Job Application Recommendation System
This project is a simple web application that recommends jobs based on the content of an uploaded resume. It uses Python, Flask for the backend, and various NLP libraries to process and match the resume against a predefined list of job descriptions.

Features
Upload resumes in PDF or DOCX format.

Uses NLP (TF-IDF and Cosine Similarity) to calculate a match score.

Displays a ranked list of relevant job recommendations.

Clean, modern, and responsive UI with a dark/light theme toggle.

Project Structure
.
├── app.py              # Main Flask application file
├── templates/
│   └── index.html      # Frontend HTML file
├── uploads/            # Temporary folder for uploaded files (created automatically)
├── requirements.txt    # Python dependencies
└── README.md           # This file

Setup and Installation
Follow these steps to get the application running on your local machine.

Prerequisites
Python 3.7 or higher

pip (Python package installer)

1. Clone the Repository
First, download the project files or clone the repository to your local machine.

2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

# Navigate to the project directory
cd path/to/your/project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
Install all the required Python packages using the requirements.txt file.

pip install -r requirements.txt

4. Download NLTK Data
The application uses NLTK for text processing. The app.py script will attempt to download the necessary data (punkt, stopwords, wordnet) on first run. If this fails due to network or permission issues, you can download them manually by running a Python shell:

python

Then, inside the Python interpreter:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
exit()

How to Run the Application
With the setup complete, you can now run the Flask application.

python app.py

You should see output similar to this in your terminal:

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
Press CTRL+C to quit

Open your web browser and navigate to the address http://127.0.0.1:5000.

How to Use the App
Click the "Click to upload" area or drag and drop your resume file (PDF or DOCX).

Click the "Get Recommendations" button.

The system will process your resume and display a list of top job matches below.