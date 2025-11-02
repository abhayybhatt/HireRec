import pandas as pd
import os

# --- Hardcoded Job Descriptions Dataset (Fallback) ---
# This list is used if 'jobs.csv' is not found or fails to load.
fallback_job_descriptions = [
    {
        "id": 1,
        "title": "Senior Python Developer",
        "company": "Tech Innovations Inc.",
        "description": "Seeking an expert Senior Python Developer with over 7 years of experience in building scalable, high-performance web applications and backend systems. Must be proficient in Django, Flask, FastAPI, and RESTful API design. Deep experience with cloud platforms like AWS (EC2, S3, Lambda) or Azure is a major plus. Strong understanding of database design, ORMs like SQLAlchemy, and both SQL (PostgreSQL) and NoSQL (MongoDB) databases is required. The ideal candidate will have hands-on experience with machine learning libraries such as Scikit-learn, Pandas, and TensorFlow."
    },
    {
        "id": 2,
        "title": "Frontend Developer (React & TypeScript)",
        "company": "Creative Solutions LLC",
        "description": "We are looking for a skilled Frontend Developer specializing in React.js to build beautiful, responsive user interfaces. The role involves translating UI/UX design wireframes into high-quality code. Key skills include mastery of HTML5, CSS3, JavaScript (ES6+), and React. Experience with state management libraries like Redux, Redux Toolkit, or Zustand is essential. Familiarity with modern frontend tools like Vite, Webpack, Tailwind CSS, and version control (Git) is also required. Experience with TypeScript is a strong plus. Candidates with full-stack MERN experience are highly encouraged to apply."
    },
    {
        "id": 3,
        "title": "Data Scientist (NLP Focus)",
        "company": "Data Insights Corp.",
        "description": "Join our advanced analytics team as a Data Scientist to analyze large text datasets and generate valuable insights. Required skills include expert-level Python, R, SQL, and experience with data visualization tools like Matplotlib, Seaborn, or Tableau. Strong background in statistical analysis, machine learning algorithms (regression, classification, clustering), and specialized NLP techniques (sentiment analysis, topic modeling) is necessary. Experience with libraries like NLTK, spaCy, and Hugging Face is critical. A Master's or PhD in Computer Science or a quantitative field is preferred."
    },
    {
        "id": 4,
        "title": "DevOps Engineer (Kubernetes & CI/CD)",
        "company": "Cloud Masters Co.",
        "description": "We need an experienced DevOps Engineer to design, manage, and scale our CI/CD pipelines and cloud infrastructure. Extensive experience with Docker, Kubernetes, Jenkins, and cloud services (AWS, GCP) is mandatory. The candidate must be proficient in scripting languages like Bash or Python. Deep knowledge of infrastructure as code (IaC) with tools like Terraform or Ansible is highly desirable. Experience with monitoring tools like Prometheus and Grafana is a plus."
    },
    {
        "id": 5,
        "title": "Junior Software Engineer (Java/Spring)",
        "company": "StartUp Hub Ventures",
        "description": "Exciting opportunity for a Junior Software Engineer to join a fast-growing startup. The candidate should have a solid foundation in Java and the Spring Boot framework. A strong understanding of object-oriented programming (OOP), data structures, and algorithms is key. Experience with building REST APIs and working with databases like MySQL or PostgreSQL is expected. Eagerness to learn new technologies and work in a collaborative agile environment is crucial."
    },
    {
        "id": 6,
        "title": "Natural Language Processing Engineer",
        "company": "AI Frontiers Lab",
        "description": "We are hiring a dedicated NLP Engineer to develop and deploy cutting-edge language models. The role requires deep knowledge of NLP techniques, including text preprocessing, feature extraction (TF-IDF, word embeddings like Word2Vec), and transformer models like BERT or GPT. Proficiency in Python and libraries like PyTorch, NLTK, spaCy, and Hugging Face Transformers is essential. A strong background in machine learning, deep learning, and software engineering is a must."
    },
    {
        "id": 7,
        "title": "UI/UX Designer",
        "company": "Pixel Perfect Designs",
        "description": "We are looking for a creative UI/UX Designer to create engaging and user-friendly interfaces for our web and mobile applications. Responsibilities include user research, creating wireframes, storyboards, sitemaps, and screen flows. Proficiency in design tools such as Figma, Sketch, or Adobe XD is required. A strong portfolio showcasing your design process and an understanding of user-centered design principles are essential."
    },
    {
        "id": 8,
        "title": "Mobile App Developer (iOS/Swift)",
        "company": "AppCrafters",
        "description": "Join our mobile team as an iOS Developer to build and maintain our suite of applications. You must have strong proficiency in Swift and the iOS SDK. Experience with frameworks like SwiftUI and UIKit is required. A deep understanding of mobile development life cycle, RESTful APIs, and Git is necessary. Experience with Core Data, Core Animation, and publishing applications on the App Store is a big plus."
    },
    {
        "id": 9,
        "title": "Database Administrator (DBA)",
        "company": "Secure Data Solutions",
        "description": "We are seeking a Database Administrator to be responsible for the performance, integrity, and security of our databases. The role involves database troubleshooting, performance tuning, and ensuring data remains consistent and is clearly defined. Strong experience with SQL and database systems like PostgreSQL, MySQL, or MS SQL Server is required. Knowledge of backup and recovery procedures and data security is critical."
    },
    {
        "id": 10,
        "title": "Product Manager",
        "company": "Innovate Tech",
        "description": "We are hiring a Product Manager to guide the success of our products and lead the cross-functional team that is responsible for improving it. This role involves defining the product vision, strategy, and roadmap. You will work closely with engineering, marketing, and sales teams. Excellent communication skills and experience in a product management or related role in the tech industry are required. Experience with Agile development methodologies is a must."
    }
]

def load_job_data(csv_path='jobs.csv'):
    """
    Loads job data from a CSV file.
    Renames columns to match the application's expected format.
    Falls back to a hardcoded list if the file is not found or is invalid.
    """
    try:
        # Check if the file exists before trying to read
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"'{csv_path}' not found.")

        jobs_df = pd.read_csv(csv_path)
        
        # Define potential column name variations from different Kaggle datasets
        column_mapping = {
            'job_title': 'title',       # From postings.csv
            'job_summary': 'description', # From postings.csv
            'company_name': 'company',    # From postings.csv
            'Job Title': 'title',       # From Software_Developer_Jobs.csv
            'Job Description': 'description', # From Software_Developer_Jobs.csv
            'Company': 'company'        # From Software_Developer_Jobs.csv
        }
        
        jobs_df.rename(columns=column_mapping, inplace=True)

        # Check if the essential columns ('title', 'company', 'description') exist after renaming
        required_columns = ['title', 'company', 'description']
        if not all(col in jobs_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in jobs_df.columns]
            raise ValueError(f"CSV file is missing required columns: {missing}")

        # Drop rows where the essential columns have missing data
        jobs_df = jobs_df[required_columns].dropna()

        # Convert the DataFrame to the list of dictionaries format
        job_descriptions = jobs_df.to_dict('records')
        
        if len(job_descriptions) == 0:
            print("WARNING: 'jobs.csv' was loaded but resulted in 0 valid job entries. Falling back to hardcoded data.")
            return fallback_job_descriptions

        print(f"Successfully loaded {len(job_descriptions)} jobs from {csv_path}")
        return job_descriptions

    except Exception as e:
        print(f"WARNING: Could not load '{csv_path}' ({e}). Falling back to hardcoded job descriptions.")
        return fallback_job_descriptions