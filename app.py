import os
import nltk
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

# --- Import from our new 'core' modules ---
from core.data_loader import load_job_data
from core.parser import parse_pdf, parse_docx
# --- FIX: Correct function names to match recommender.py ---
from core.recommender import (
    preprocess_text,
    get_tfidf_recommendations,
    get_spacy_recommendations
)
# --- END OF FIX ---

# --- NLTK Setup ---
print("Checking for NLTK data packages...")
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
print("NLTK checks complete.")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Job Data ONCE ---
print("Loading job data...")
try:
    job_descriptions = load_job_data('jobs.csv')
    print(f"Successfully loaded {len(job_descriptions)} jobs.")
except Exception as e:
    print(f"Error loading job data: {e}")
    job_descriptions = [] 

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handles the resume upload, processes it, and returns job recommendations
    from both TF-IDF and SpaCy models.
    """
    if not job_descriptions:
        return jsonify({"error": "Job dataset is not loaded. Please check server logs."}), 500
        
    # 1. File Handling
    if 'resume' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(file_path)

        # 2. Text Extraction (using our parser module)
        raw_text = ""
        if filename.lower().endswith('.pdf'):
            raw_text = parse_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            raw_text = parse_docx(file_path)
        else:
            return jsonify({"error": "Unsupported file type. Please upload a PDF or DOCX file."}), 400
        
        if not raw_text.strip():
             return jsonify({"error": "Could not extract text from the resume. The file might be empty or corrupted."}), 400

    except Exception as e:
        return jsonify({"error": f"Error parsing file: {e}"}), 500
    finally:
         if os.path.exists(file_path):
            os.remove(file_path) # Clean up the uploaded file

    # 3. Generate Recommendations (using our recommender module)
    try:
        # --- FIX: We must pass the RAW text to the spaCy function ---
        raw_resume_text = raw_text
        # And the PREPROCESSED text to the TF-IDF function
        processed_resume = preprocess_text(raw_resume_text)
        # --- END OF FIX ---

        if not processed_resume.strip():
            return jsonify({"error": "Your resume does not contain enough relevant keywords after processing."}), 400

        # Get results from Model 1: TF-IDF
        tfidf_recs = get_tfidf_recommendations(processed_resume, job_descriptions, top_n=20)
        
        # Get results from Model 2: SpaCy
        spacy_recs = get_spacy_recommendations(raw_resume_text, job_descriptions, top_n=20)

        # Return both sets of recommendations
        return jsonify({
            "tfidf": tfidf_recs,
            "spacy": spacy_recs
        })
        
    except ValueError as e:
        # Handle errors if resume is empty after processing
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Catch any other unexpected errors
        print(f"!!! SERVER CRASH: {e}") # Added for better debugging
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# --- Main Entry Point ---
if __name__ == '__main__':
    app.run(debug=True)