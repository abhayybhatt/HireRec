import os
import nltk
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

# --- Import from our new 'core' modules ---
from core.data_loader import load_job_data
from core.parser import parse_resume
from core.recommender import (
    get_recommendations_tfidf, 
    get_recommendations_spacy,
    preprocess_text_nltk, # Needed for NLTK package downloads
    preprocess_text_spacy # Needed for spaCy model check
)

# --- NLTK & SpaCy Setup ---
# We run the downloads/checks here in the main app file
# to ensure they are available when the server starts.
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

# Trigger the spaCy model check from the recommender module
print("Checking for spaCy model...")
preprocess_text_spacy("test") # This will trigger the print message if model is missing
print("SpaCy check complete.")


# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Job Data ONCE ---
# Load the job descriptions into memory when the app starts.
print("Loading job data...")
try:
    job_descriptions = load_job_data('jobs.csv')
    print(f"Successfully loaded {len(job_descriptions)} jobs.")
except Exception as e:
    print(f"Error loading job data: {e}")
    job_descriptions = [] # Start with an empty list if loading fails

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
        raw_text = parse_resume(file_path, file.filename)
        
        if not raw_text.strip():
             return jsonify({"error": "Could not extract text from the resume. The file might be empty or corrupted."}), 400

    except Exception as e:
        return jsonify({"error": f"Error parsing file: {e}"}), 500
    finally:
         if os.path.exists(file_path):
            os.remove(file_path) # Clean up the uploaded file

    # 3. Generate Recommendations (using our recommender module)
    try:
        # Get results from Model 1: TF-IDF
        tfidf_recs = get_recommendations_tfidf(raw_text, job_descriptions)
        
        # Get results from Model 2: SpaCy
        spacy_recs = get_recommendations_spacy(raw_text, job_descriptions)

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
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# --- Main Entry Point ---
if __name__ == '__main__':
    app.run(debug=True)