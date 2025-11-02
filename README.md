HireRec: A Job Application Recommendation System
A dual-model NLP application built with Python and Flask that intelligently matches a user's resume to a large dataset of job postings.

This project implements and compares two distinct natural language processing models to provide job recommendations:

TF-IDF (Keyword-Based): A classic NLP model that matches exact keywords and phrases between the resume and job descriptions.

Semantic Search (spaCy): A modern, vector-based model that understands the contextual meaning of words (e.g., it knows "React" and "Angular" are both frontend frameworks).

The frontend allows you to upload a resume and instantly see the results from both models side-by-side in a tabbed view, making it an excellent tool for comparing the effectiveness of these two different AI approaches.

Features
Dual Recommendation Models: Get results from both TF-IDF and spaCy (semantic) models at the same time.

Tabbed Interface: Easily switch between model results to compare recommendations.

Dynamic File Upload: Supports .pdf and .docx resume formats.

Large Dataset: Uses a CSV of thousands of real job postings (not included in repo).

Light/Dark Mode: Includes a theme-switching toggle for the UI.

Modular Backend: Code is structured into a clean core/ directory for maintainability.

Automated Tests: Includes pytest unit tests for the core recommendation logic.

Project Structure
job-recommender/
├── venv/
├── core/
│   ├── __init__.py           # Makes 'core' a Python package
│   ├── data_loader.py        # Loads and manages the CSV data
│   ├── parser.py             # Handles PDF/DOCX resume parsing
│   └── recommender.py        # Holds both TF-IDF & SpaCy models
│
├── templates/
│   └── index.html            # Frontend user interface
│
├── tests/
│   └── test_recommender.py   # Automated unit tests
│
├── uploads/                  # Temporary folder for file uploads
│
├── app.py                    # Main Flask app (handles routes)
├── jobs.csv                  # (Not included) Your dataset of job postings
├── README.md                 # This file
└── requirements.txt          # Python packages

Setup and Installation
Follow these steps to get the project running locally.

1. Clone the Repository

git clone [your-github-repo-url]
cd job-recommender

2. Create and Activate a Virtual Environment

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
This project requires several packages. Make sure you have Microsoft C++ Build Tools installed if you are on Windows, as scikit-learn needs it to compile.

# Upgrade pip (recommended)
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

4. Download the spaCy Language Model
The semantic model requires a small language model from spaCy.

python -m spacy download en_core_web_sm

5. Get the Job Dataset
This project is designed to work with a large CSV of job postings.

Download a dataset (e.g., from Kaggle).

Rename the file to jobs.csv.

Place jobs.csv in the root of the project folder.

Note: The code is set to look for columns named job_title, company, and job_summary. If your CSV uses different names, you must update the column_mapping dictionary in core/data_loader.py.

How to Run
1. Run the Flask Application
With your virtual environment active, start the backend server:

python app.py
Your server will be running at http://127.0.0.1:5000.

2. Open the Frontend
Open the templates/index.html file directly in your web browser (e.g., Chrome, Firefox).

You can now upload your resume and get recommendations.

3. Run Automated Tests
To verify that the core logic is working correctly, you can run the built-in tests using pytest command:

pytest
You should see all tests passing in your terminal.