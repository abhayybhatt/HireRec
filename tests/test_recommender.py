import pytest
import spacy
from core.recommender import (
    preprocess_text,
    get_tfidf_recommendations,
    get_spacy_recommendations
)

# --- Fixtures: Setup code that runs before tests ---

@pytest.fixture(scope="module")
def spacy_model():
    """Loads the spaCy model once for all tests in this module."""
    try:
        # Load the medium model, which includes vectors
        nlp = spacy.load("en_core_web_md")
        return nlp
    except OSError:
        print("Test spaCy 'en_core_web_md' model not found. Please run:")
        print("python -m spacy download en_core_web_md")
        return None

@pytest.fixture
def sample_resume():
    """Provides a sample preprocessed resume text for testing."""
    raw = "Experienced React developer with skills in JavaScript, HTML, and CSS. Worked on MERN stack projects."
    return preprocess_text(raw)

@pytest.fixture
def sample_job_list():
    """Provides a sample list of job descriptions for testing."""
    return [
        {
            "title": "Frontend Developer",
            "company": "TechA",
            "description": "We need a React developer. JavaScript, HTML, CSS are a must. MERN stack a plus."
        },
        {
            "title": "Backend Python Developer",
            "company": "DataCo",
            "description": "Seeking Python expert for data pipelines. No JavaScript required."
        },
        {
            "title": "UI Designer",
            "company": "DesignFirm",
            "description": "Focus on design, figma, and user experience. No coding."
        }
    ]

# --- Test Cases ---

def test_preprocess_text():
    """Tests that text preprocessing correctly cleans, tokenizes, and lemmatizes."""
    raw = "This is a TEST string!! with numbers 123 and stopwords... running."
    processed = preprocess_text(raw)
    assert processed == "test string number stopword running"

def test_preprocess_empty_text():
    """Tests that preprocessing an empty string returns an empty string."""
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""

def test_tfidf_recommendations(sample_resume, sample_job_list):
    """Tests the TF-IDF model for basic keyword matching."""
    recommendations = get_tfidf_recommendations(sample_resume, sample_job_list, top_n=3)
    
    # The first job ("Frontend Developer") should be the top match
    assert len(recommendations) == 3
    assert recommendations[0]['title'] == "Frontend Developer"
    assert recommendations[0]['score'] > 0
    
    # The third job ("UI Designer") should have a very low score
    assert recommendations[2]['title'] == "UI Designer"
    assert recommendations[2]['score'] >= 0 # (could be 0)

def test_spacy_recommendations(spacy_model, sample_resume, sample_job_list):
    """Tests the spaCy (semantic) model for contextual matching."""
    if spacy_model is None:
        pytest.skip("spaCy 'en_core_web_md' model not found. Skipping test.")
        
    recommendations = get_spacy_recommendations(sample_resume, sample_job_list, top_n=3)
    
    # The first job ("Frontend Developer") should be the top match
    assert len(recommendations) == 3
    assert recommendations[0]['title'] == "Frontend Developer"
    assert recommendations[0]['score'] > 0
    
    # The other jobs should have lower scores
    assert recommendations[0]['score'] > recommendations[1]['score']
    assert recommendations[1]['score'] > recommendations[2]['score']

def test_empty_resume_tfidf():
    """Tests that the TF-IDF model handles an empty resume string."""
    recommendations = get_tfidf_recommendations("", [], top_n=3)
    assert recommendations == []

def test_empty_resume_spacy(spacy_model):
    """Tests that the spaCy model handles an empty resume string."""
    if spacy_model is None:
        pytest.skip("spaCy 'en_core_web_md' model not found. Skipping test.")
        
    recommendations = get_spacy_recommendations("", [], top_n=3)
    assert recommendations == []