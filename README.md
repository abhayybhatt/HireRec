
---

# HireRec: A Job Application Recommendation System

A dual-model NLP application built with **Python** and **Flask** that intelligently matches a user's resume to a large dataset of job postings.

---

## 🧠 Overview

This project implements and compares two distinct natural language processing (NLP) models to provide job recommendations:

* **TF-IDF (Keyword-Based):** A classic NLP model that matches exact keywords and phrases between the resume and job descriptions.
* **Semantic Search (spaCy):** A modern, vector-based model that understands the contextual meaning of words (e.g., it knows “React” and “Angular” are both frontend frameworks).

The frontend allows you to upload a resume and instantly see results from both models **side-by-side in a tabbed view**, making it an excellent tool for comparing the effectiveness of these two different AI approaches.

---

## ✨ Features

* **Dual Recommendation Models:** Get results from both TF-IDF and spaCy (semantic) models simultaneously.
* **Tabbed Interface:** Easily switch between model results to compare recommendations.
* **Dynamic File Upload:** Supports `.pdf` and `.docx` resume formats.
* **Large Dataset:** Uses a CSV of thousands of real job postings (not included in repo).
* **Light/Dark Mode:** Includes a theme-switching toggle for the UI.
* **Modular Backend:** Code is structured into a clean `core/` directory for maintainability.
* **Automated Tests:** Includes pytest unit tests for the core recommendation logic.

---

## 📂 Project Structure


<img width="351" height="434" alt="image" src="https://github.com/user-attachments/assets/1223f4d2-5fee-462c-9fe4-0cfc1e4ecea1" />


---

> 💡 **Tip:** Make sure your `.gitignore` includes `venv/` and `__pycache__/` so your virtual environment and cache files don’t get pushed to GitHub.


## ⚙️ Setup and Installation

Follow these steps to get the project running locally.

### 1. Clone the Repository

git clone [your-github-repo-url]
cd job-recommender

---

### 2. Create and Activate a Virtual Environment

#### 🪟 Windows

python -m venv venv
venv\Scripts\activate

#### 🐧 macOS/Linux

python3 -m venv venv
source venv/bin/activate

---

### 3. Install Dependencies

This project requires several Python packages to function correctly.
If you are on **Windows**, make sure you have **Microsoft C++ Build Tools** installed (as scikit-learn needs it to compile).

#### 🔹 Upgrade pip (recommended)

python -m pip install --upgrade pip

#### 🔹 Install all required packages

pip install -r requirements.txt

---

### 4. Download the spaCy Language Model

The semantic model uses the small English language model from **spaCy**.

python -m spacy download en_core_web_sm

---

### 5. Get the Job Dataset

This project is designed to work with a large CSV of job postings.

1. Download a dataset (e.g., from Kaggle).
2. Rename the downloaded file to **jobs.csv**.
3. Place jobs.csv in the **root** of the project folder.

**Note:**
The application expects the CSV to include the columns:
`job_title`, `company`, and `job_summary`.
If your CSV has different column names, update the `column_mapping` dictionary in `core/data_loader.py`.

---

## 🚀 How to Run

### 1. Start the Flask Application

With your virtual environment active, run:

python app.py

Your server will start and be available at:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 2. Launch the Frontend

Open the following file directly in your browser (e.g., Chrome, Firefox):

templates/index.html

You can now upload your **resume** and get **job recommendations** instantly.

---

### 3. Run Automated Tests

To verify that the core logic is working correctly, run the built-in tests using **pytest**:

pytest

You should see all tests passing in your terminal.

---

## 🧩 Troubleshooting

If you encounter errors during setup or model loading:

* Ensure your **virtual environment** is activated.
* Double-check that all dependencies from requirements.txt are installed.
* Verify that the **CSV file path** is correct and named jobs.csv.
* For spaCy-related errors, rerun:
  python -m spacy download en_core_web_sm

---

## 🧠 Technologies Used

| Category       | Technology                                        |
| -------------- | ------------------------------------------------- |
| **Backend**    | Python, Flask                                     |
| **NLP Models** | TF-IDF, spaCy (Semantic Search)                   |
| **Libraries**  | scikit-learn, pandas, numpy, PyMuPDF, python-docx |
| **Frontend**   | HTML, CSS, JavaScript                             |
| **Testing**    | pytest                                            |
| **Dataset**    | CSV (job postings dataset)                        |


## 🧪 Example Usage

1. Start the Flask server:
   python app.py

2. Visit the web app in your browser:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. Upload your **resume** (.pdf or .docx).

4. Choose a model (TF-IDF or spaCy) to get job recommendations.

5. Compare both models’ recommendations side-by-side.

---

## 🧰 Future Improvements

* Add more advanced embedding models (e.g., Sentence Transformers, BERT).
* Integrate database storage for uploaded resumes and job results.
* Add user authentication and personalized dashboards.
* Expand UI with filters (location, salary range, skill match %).

---

## 📄 License

This project is open-source and available under the **MIT License**.
You are free to use, modify, and distribute it with proper attribution.

---
