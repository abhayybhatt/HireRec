import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Models (Done once on startup) ---

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- FIX: Load the 'medium' model (md) which includes word vectors ---
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("="*40)
    print("ERROR: spaCy 'en_core_web_md' model not found.")
    print("Please run this command in your terminal:")
    print("python -m spacy download en_core_web_md")
    print("="*40)
    nlp = None

# --- Core Functions ---

def preprocess_text(text):
    """
    Cleans and preprocesses text FOR TF-IDF ONLY:
    1. Removes punctuation/numbers
    2. Converts to lowercase
    3. Tokenizes
    4. Removes stopwords
    5. Lemmatizes
    """
    if text is None:
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2
    ]
    return " ".join(processed_tokens)

def get_tfidf_recommendations(processed_resume_text, job_list, top_n=100):
    """
    Generates job recommendations using the TF-IDF (keyword-based) model.
    This function EXPECTS preprocessed resume text.
    """
    # Create a corpus of all documents
    jobs_processed = [preprocess_text(job['description']) for job in job_list]
    corpus = [processed_resume_text] + jobs_processed
    
    # Initialize and fit the vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 3)) # Use 1, 2, and 3-word phrases
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Check for empty vocabulary
    if tfidf_matrix.shape[1] == 0:
        return []

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    recommendations = []
    for i, score in enumerate(cosine_similarities):
        # We need to add the original job description to the dict for the funnel
        recommendations.append({
            "title": job_list[i]['title'],
            "company": job_list[i]['company'],
            "description": job_list[i]['description'], # Add ORIGINAL description for spaCy
            "score": round(score * 100, 2)
        })
    
    # Sort and return the top N
    sorted_recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    return sorted_recommendations[:top_n]


def get_spacy_recommendations(raw_resume_text, job_list, top_n=20):
    """
    Generates job recommendations using the spaCy (semantic-based) model.
    This function EXPECTS RAW resume text.
    
    --- This is a 2-stage "funnel" system ---
    1. It first uses the fast TF-IDF model to find the Top 100 most relevant jobs.
    2. It then uses the "smarter" spaCy model to re-rank only those 100 jobs.
    """
    
    # --- LOGIC FIX: This check MUST be at the very top. ---
    if nlp is None:
        print("CRITICAL ERROR: spaCy 'en_core_web_md' model is not loaded. Cannot generate spaCy recommendations.")
        return [{"title": "Error: Model Not Found", "company": "Run 'python -m spacy download en_core_web_md' in terminal", "score": 0}]
    # --- END OF FIX ---

    if not raw_resume_text:
        return []

    print("Running spaCy funnel: Finding Top 100 TF-IDF candidates...")
    
    # --- STAGE 1: Get Top 100 candidates from TF-IDF ---
    # We must preprocess the raw resume text here *just* for the TF-IDF function.
    processed_resume_for_tfidf = preprocess_text(raw_resume_text)
    if not processed_resume_for_tfidf:
         return [] # Resume was empty after preprocessing
         
    tfidf_candidates = get_tfidf_recommendations(processed_resume_for_tfidf, job_list, top_n=100)
    
    if not tfidf_candidates:
        return []

    print(f"Re-ranking {len(tfidf_candidates)} candidates with spaCy...")

    # --- STAGE 2: Re-rank the Top 100 with spaCy (semantic) ---
    recommendations = []
    
    # --- BUG FIX: Use the RAW resume text for the main spaCy doc ---
    # We must disable 'parser' and 'ner' for speed when just getting vectors
    resume_doc = nlp(raw_resume_text, disable=['parser', 'ner'])
    if not resume_doc.has_vector or not resume_doc.vector_norm:
         print("Warning: Resume has no vector. spaCy results may be poor.")
         return [{"title": "Error", "company": "Could not process resume vector.", "score": 0}]


    # Process only the 100 candidates
    for job in tfidf_candidates:
        
        # --- BUG FIX: Use the RAW job description for the job doc ---
        # We must disable 'parser' and 'ner' for speed
        # Also ensure description is a string
        job_desc_text = str(job['description']) if job['description'] is not None else ""
        if not job_desc_text:
            continue
            
        job_doc = nlp(job_desc_text, disable=['parser', 'ner'])
        
        if not job_doc.has_vector or not job_doc.vector_norm:
            continue # Skip this job if it has no vector

        # Calculate semantic similarity
        score = resume_doc.similarity(job_doc)
        
        recommendations.append({
            "title": job['title'],
            "company": job['company'],
            # --- SYNTAX ERROR FIX: Changed 1T00 to 100 ---
            "score": round(score * 100, 2)
        })

    # Sort and return the final Top N
    sorted_recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    print("spaCy re-ranking complete.")
    return sorted_recommendations[:top_n]

