import streamlit as st
import pdfplumber
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Category-wise keywords for ATS scoring
keywords_dict = {
    "HR": ["recruitment", "employee", "training", "hiring", "management", "payroll", "interview", "team"],
    "Data Science": ["python", "machine learning", "data", "model", "analytics", "pandas", "numpy", "matplotlib", "ai", "visualization"],
    "Software Engineer": ["python", "java", "api", "git", "sql", "debugging", "development", "backend", "frontend", "react"],
    "Designer": ["photoshop", "illustrator", "ui", "ux", "figma", "adobe", "creativity", "branding", "design"],
    "Finance": ["accounting", "budget", "tax", "audit", "finance", "excel", "reporting", "profit", "loss"],
}

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Calculate ATS score
def calculate_ats_score(resume_text, job_keywords):
    resume_words = clean_text(resume_text).split()
    matched = [word for word in resume_words if word in job_keywords]
    score = (len(set(matched)) / len(set(job_keywords))) * 100 if job_keywords else 0
    return round(score, 2), matched

# Streamlit UI
st.set_page_config(page_title="ATS Score Checker", page_icon="📄", layout="centered")
st.title("📄 ATS Score Checker (Auto Mode)")
st.write("Upload your resume (PDF) — the system will automatically detect your domain and calculate your ATS score.")

uploaded_file = st.file_uploader("📂 Upload Resume (PDF format)", type=["pdf"])

if uploaded_file:
    if st.button("🔍 Check ATS Score"):
        st.info("Extracting text and analyzing resume...")
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_text(resume_text)

        # Predict resume category
        X_vec = vectorizer.transform([cleaned_text])
        predicted_category = model.predict(X_vec)[0]
        st.write(f"🧠 **Detected Resume Category:** `{predicted_category}`")

        # Calculate ATS score
        job_keywords = keywords_dict.get(predicted_category, [])
        if not job_keywords:
            st.warning("No predefined keywords available for this category.")
        else:
            score, matched = calculate_ats_score(resume_text, job_keywords)
            st.success(f"✅ **ATS Score: {score}%**")
            st.write("**Matched Keywords:**", ", ".join(set(matched)))
