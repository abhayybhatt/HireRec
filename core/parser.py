import PyPDF2
import docx
import os

def parse_pdf(file_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            # Add a space between pages to avoid merging words
            text += (page.extract_text() or "") + " "
    return text

def parse_docx(file_path):
    """
    Extracts text from a DOCX file.
    """
    document = docx.Document(file_path)
    return "\n".join([para.text for para in document.paragraphs])

def parse_resume(file_path):
    """
    Parses an uploaded resume file (PDF or DOCX) and extracts the raw text.
    
    Args:
        file_path (str): The full path to the uploaded file.

    Returns:
        str: The extracted raw text from the document.
        
    Raises:
        ValueError: If the file type is not supported.
    """
    # Get the file extension
    filename = os.path.basename(file_path)
    
    try:
        if filename.lower().endswith('.pdf'):
            return parse_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            return parse_docx(file_path)
        else:
            # This should be caught by the Flask app first, but serves as a safeguard.
            raise ValueError(f"Unsupported file type: {filename}. Please upload a PDF or DOCX file.")
    except Exception as e:
        # Catch potential corruption or parsing errors
        print(f"Error parsing file {filename}: {e}")
        raise ValueError(f"Could not parse file. It may be corrupted or password-protected.")