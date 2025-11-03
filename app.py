import streamlit as st
import pdfplumber
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load model & vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to calculate ATS score
def calculate_ats_score(resume_text, job_keywords):
    resume_words = clean_text(resume_text).split()
    job_words = clean_text(job_keywords).split()

    matched = [word for word in resume_words if word in job_words]
    score = (len(set(matched)) / len(set(job_words))) * 100 if job_words else 0
    return round(score, 2), matched

# Streamlit UI
st.set_page_config(page_title="ATS Score Checker", page_icon="📄", layout="centered")

st.title("📄 ATS Score Checker")
st.write("Upload your resume and get an ATS match score based on job description or keywords.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_keywords = st.text_area("Paste Job Description or Keywords")

if uploaded_file and job_keywords:
    if st.button("Check ATS Score"):
        st.info("Extracting text and analyzing...")
        resume_text = extract_text_from_pdf(uploaded_file)
        score, matched = calculate_ats_score(resume_text, job_keywords)

        st.success(f"✅ Your ATS Match Score: **{score}%**")
        st.write("**Matched Keywords:** ", ", ".join(set(matched)) if matched else "No significant matches found.")
        if score < 50:
            st.warning("⚠️ Try adding more relevant keywords to improve your score.")
        elif score < 75:
            st.info("🙂 Good! But you can still add a few more relevant terms.")
        else:
            st.balloons()
            st.success("🎉 Great! Your resume is well optimized for this job.")
