import os
import re
import nltk
import docx
import PyPDF2
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK Setup ---
# Download necessary NLTK data (only needs to be done once)
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

# Initialize NLTK components
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Hardcoded Job Descriptions Dataset ---
# As per the requirements, using a sample dataset directly in the script.
job_descriptions = [
    {
        "id": 1,
        "title": "Senior Python Developer",
        "company": "Tech Innovations Inc.",
        "description": "Seeking a Senior Python Developer with over 5 years of experience in building scalable web applications. Must be proficient in Django, Flask, and RESTful APIs. Experience with cloud platforms like AWS or Azure is a plus. Strong understanding of data structures, algorithms, and software design principles is required. The ideal candidate will have experience with machine learning libraries such as Scikit-learn and TensorFlow."
    },
    {
        "id": 2,
        "title": "Frontend Developer (React)",
        "company": "Creative Solutions",
        "description": "We are looking for a skilled Frontend Developer specializing in React.js. The role involves creating modern, responsive user interfaces. Key skills include proficiency in HTML, CSS, JavaScript, and React. Experience with state management libraries like Redux or Zustand is essential. Familiarity with Tailwind CSS and version control (Git) is also required."
    },
    {
        "id": 3,
        "title": "Data Scientist",
        "company": "Data Insights Corp.",
        "description": "Join our team as a Data Scientist to analyze large datasets and generate valuable insights. Required skills include Python, R, SQL, and experience with data visualization tools like Matplotlib or Tableau. Strong background in statistical analysis, machine learning algorithms (like regression, classification), and NLP techniques is necessary. A Master's or PhD in a quantitative field is preferred."
    },
    {
        "id": 4,
        "title": "DevOps Engineer",
        "company": "Cloud Masters",
        "description": "We need a DevOps Engineer to manage our CI/CD pipelines and cloud infrastructure. Experience with Docker, Kubernetes, Jenkins, and cloud services (AWS, GCP) is mandatory. The candidate should be proficient in scripting languages like Bash or Python. Knowledge of infrastructure as code (IaC) with tools like Terraform or Ansible is highly desirable."
    },
    {
        "id": 5,
        "title": "Junior Software Engineer",
        "company": "StartUp Hub",
        "description": "Exciting opportunity for a Junior Software Engineer to join a fast-growing startup. The candidate should have a solid foundation in at least one programming language like Java, C++, or Python. Understanding of object-oriented programming, data structures, and algorithms is key. Eagerness to learn new technologies and work in an agile environment is crucial."
    },
    {
        "id": 6,
        "title": "Natural Language Processing Engineer",
        "company": "AI Frontiers",
        "description": "We are hiring an NLP Engineer to work on cutting-edge language models. The role requires deep knowledge of NLP techniques, including text preprocessing, feature extraction (TF-IDF, word embeddings), and models like BERT or GPT. Proficiency in Python and libraries like NLTK, spaCy, and Hugging Face Transformers is essential. A strong background in machine learning and deep learning is a must."
    }
]

# --- Core Logic Functions ---

def parse_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def parse_docx(file_path):
    """Extracts text from a DOCX file."""
    document = docx.Document(file_path)
    return "\n".join([para.text for para in document.paragraphs])

def preprocess_text(text):
    """Cleans and preprocesses text using NLP techniques."""
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2
    ]
    return " ".join(processed_tokens)

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        # 1. File Handling
        if 'resume' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        file = request.files['resume']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        # Securely save the uploaded file
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 2. Text Extraction
        raw_text = ""
        try:
            if filename.lower().endswith('.pdf'):
                raw_text = parse_pdf(file_path)
            elif filename.lower().endswith('.docx'):
                raw_text = parse_docx(file_path)
            else:
                os.remove(file_path)
                return jsonify({"error": "Unsupported file type. Please upload a PDF or DOCX file."}), 400
        except Exception as e:
            os.remove(file_path)
            return jsonify({"error": f"Error parsing file: {e}"}), 500
        finally:
             if os.path.exists(file_path):
                os.remove(file_path) # Clean up the uploaded file

        if not raw_text.strip():
             return jsonify({"error": "Could not extract text from the resume. The file might be empty or corrupted."}), 400


        # 3. NLP Preprocessing
        resume_processed = preprocess_text(raw_text)

        if not resume_processed.strip():
            return jsonify({"error": "Your resume does not contain enough relevant keywords after processing. Please use a more detailed resume."}), 400

        jobs_processed = [preprocess_text(job['description']) for job in job_descriptions]

        # 4. Recommendation Engine (TF-IDF & Cosine Similarity)
        try:
            corpus = [resume_processed] + jobs_processed
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Check if the vectorizer produced any features. If not, the resume is likely too generic.
            if tfidf_matrix.shape[1] == 0:
                return jsonify({"error": "Could not generate recommendations. The resume may not contain enough relevant keywords."}), 400

            # Calculate cosine similarity between the resume (index 0) and all jobs
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # 5. Generate Recommendations
            recommendations = []
            for i, score in enumerate(cosine_similarities):
                recommendations.append({
                    "title": job_descriptions[i]['title'],
                    "company": job_descriptions[i]['company'],
                    "score": round(score * 100, 2) # Convert to percentage
                })
            
            # Sort recommendations by score in descending order
            sorted_recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)

            return jsonify(sorted_recommendations)
        except Exception as e:
            # Catch any other unexpected errors during the ML process
            return jsonify({"error": f"An unexpected error occurred during the recommendation process: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)


