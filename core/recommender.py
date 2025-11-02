import re
import nltk
import spacy # New import for our second model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK Setup ---
# This ensures the necessary packages are available.
# We run this once in the main app.py, but it's good practice to
# have the components initialized here where they are used.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- SpaCy Model Loading ---
# Load the small English model for semantic similarity
# This requires running: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load('en_core_web_sm')
except IOError:
    print("SpaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None

# --- NLP Preprocessing ---

def preprocess_text_nltk(text):
    """
    Cleans and preprocesses text using NLTK for TF-IDF.
    """
    text = str(text) # Ensure text is string
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2
    ]
    return " ".join(processed_tokens)

def preprocess_text_spacy(text):
    """
    Cleans and preprocesses text using SpaCy for semantic vectors.
    Removes stop words, punctuation, and lemmatizes.
    """
    text = str(text) # Ensure text is string
    if not nlp:
        return text # Return raw text if spacy model failed to load
        
    doc = nlp(text)
    processed_tokens = [
        token.lemma_.lower() for token in doc 
        if not token.is_stop and not token.is_punct and len(token.lemma_) > 2 and token.has_vector
    ]
    return " ".join(processed_tokens)

# --- Recommendation Model 1: TF-IDF (Keyword-Based) ---

def get_recommendations_tfidf(resume_text, job_descriptions, top_n=20):
    """
    Generates job recommendations using TF-IDF and Cosine Similarity.
    """
    # Preprocess all documents
    resume_processed = preprocess_text_nltk(resume_text)
    jobs_processed = [preprocess_text_nltk(job['description']) for job in job_descriptions]

    if not resume_processed.strip():
        raise ValueError("Your resume does not contain enough relevant keywords after processing.")

    # Create the TF-IDF matrix
    corpus = [resume_processed] + jobs_processed
    # Use n-grams (1, 2, 3) to capture multi-word phrases
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError as e:
        # This can happen if all words are stop words
        print(f"Error during TF-IDF vectorization: {e}")
        raise ValueError("Could not vectorize resume, it may be empty or contain only stop words.")

    if tfidf_matrix.shape[1] == 0:
        raise ValueError("Could not generate recommendations. The resume may not contain enough relevant keywords.")

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    recommendations = []
    for i, score in enumerate(cosine_similarities):
        recommendations.append({
            "title": job_descriptions[i]['title'],
            "company": job_descriptions[i]['company'],
            "score": round(score * 100, 2)
        })

    # Sort and return the top N results
    sorted_recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    return sorted_recommendations[:top_n]

# --- Recommendation Model 2: SpaCy (Semantic-Based) ---

def get_recommendations_spacy(resume_text, job_descriptions, top_n=20):
    """
    Generates job recommendations using SpaCy's word embeddings (semantic similarity).
    """
    if not nlp:
        raise RuntimeError("SpaCy 'en_core_web_sm' model is not loaded. Please download it first.")

    # Preprocess and get the document vector for the resume
    resume_doc = nlp(preprocess_text_spacy(resume_text))
    
    if not resume_doc.has_vector or resume_doc.vector_norm == 0:
         raise ValueError("Could not generate a valid vector for the resume.")

    recommendations = []
    for job in job_descriptions:
        job_desc = job['description']
        
        # Preprocess and get the document vector for the job
        job_doc = nlp(preprocess_text_spacy(job_desc))
        
        # Calculate similarity, handling cases where a doc might be empty
        score = 0
        if job_doc.has_vector and job_doc.vector_norm != 0:
            score = resume_doc.similarity(job_doc)
        
        recommendations.append({
            "title": job['title'],
            "company": job['company'],
            "score": round(score * 100, 2)
        })

    # Sort and return the top N results
    sorted_recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    return sorted_recommendations[:top_n]