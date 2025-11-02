import pytest
from core.recommender import (
    preprocess_text_nltk, 
    preprocess_text_spacy, 
    get_recommendations_tfidf, 
    get_recommendations_spacy
)
import spacy

# --- Test Fixtures ---
# Fixtures are reusable objects for tests.
@pytest.fixture
def sample_resume():
    """A sample resume text for testing."""
    return """
    Abhay Bhatt
    Full-Stack Developer
    Passionate about MERN stack (MongoDB, Express.js, React.js, Node.js).
    Created an Airbnb clone and a news aggregator.
    Skills in REST APIs, Redux, and Tailwind CSS.
    """

@pytest.fixture
def sample_jobs():
    """A sample list of job descriptions for testing."""
    return [
        {
            "title": "Frontend Developer", 
            "company": "React World", 
            "description": "We need a React.js expert. Experience with Redux and Tailwind CSS is a must."
        },
        {
            "title": "Backend Developer", 
            "company": "Python Inc.", 
            "description": "Loves Python, Django, and Flask. No JavaScript required."
        },
        {
            "title": "Full-Stack Engineer", 
            "company": "MERN Solutions", 
            "description": "Looking for a MERN stack developer. Must know MongoDB, Express.js, React.js, and Node.js."
        }
    ]

@pytest.fixture(scope="session")
def spacy_model():
    """Load the spaCy model once for all tests."""
    try:
        return spacy.load('en_core_web_sm')
    except IOError:
        pytest.skip("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

# --- Test Cases ---

def test_preprocess_text_nltk():
    """Test the NLTK text preprocessing function."""
    text = "Hello world! This is a test, with punctuation and 123 numbers."
    processed = preprocess_text_nltk(text)
    assert processed == "hello world test punctuation number"
    assert "123" not in processed
    assert "!" not in processed
    assert "is" not in processed # Stop word removed

def test_preprocess_text_spacy(spacy_model):
    """Test the SpaCy text preprocessing function."""
    text = "Hello worlds! This is testing, with punctuation and 123 numbers."
    processed = preprocess_text_spacy(text)
    # Note: spaCy's lemmatizer might behave slightly differently, e.g., 'testing' -> 'test'
    assert processed == "hello world test punctuation number"
    assert "123" not in processed
    assert "!" not in processed
    assert "is" not in processed # Stop word removed

def test_tfidf_recommendations(sample_resume, sample_jobs):
    """Test if TF-IDF model returns sorted recommendations."""
    recs = get_recommendations_tfidf(sample_resume, sample_jobs)
    
    # We expect 3 results since we provided 3 jobs
    assert len(recs) == 3
    
    # The first result should be the "Frontend Developer" or "Full-Stack"
    # and scores should be descending.
    assert recs[0]['score'] >= recs[1]['score']
    assert recs[1]['score'] >= recs[2]['score']
    
    # Check that the MERN job is the top match
    assert recs[0]['title'] == "Full-Stack Engineer"
    assert recs[1]['title'] == "Frontend Developer"

def test_spacy_recommendations(sample_resume, sample_jobs, spacy_model):
    """Test if SpaCy model returns sorted recommendations."""
    # Ensure the model is loaded in the recommender module
    from core import recommender
    recommender.nlp = spacy_model
    
    recs = get_recommendations_spacy(sample_resume, sample_jobs)
    
    assert len(recs) == 3
    
    # Scores should be descending
    assert recs[0]['score'] >= recs[1]['score']
    assert recs[1]['score'] >= recs[2]['score']

    # SpaCy is smart about meaning. It should rank "Full-Stack" and "Frontend" high.
    assert recs[0]['title'] in ["Full-Stack Engineer", "Frontend Developer"]
    assert recs[1]['title'] in ["Full-Stack Engineer", "Frontend Developer"]
    assert recs[2]['title'] == "Backend Developer"

def test_empty_resume_tfidf(sample_jobs):
    """Test if TF-IDF model handles an empty resume gracefully."""
    empty_resume = "This is a stop word."
    # We expect a ValueError, not a crash
    with pytest.raises(ValueError):
        get_recommendations_tfidf(empty_resume, sample_jobs)

def test_empty_resume_spacy(sample_jobs, spacy_model):
    """Test if SpaCy model handles an empty resume gracefully."""
    # Ensure the model is loaded
    from core import recommender
    recommender.nlp = spacy_model
    
    empty_resume = "this is a stop word"
    # We expect a ValueError
    with pytest.raises(ValueError):
        get_recommendations_spacy(empty_resume, sample_jobs)