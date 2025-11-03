import streamlit as st
import pdfplumber
import re
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="ATS Score Checker", page_icon="📄", layout="centered")

st.title("📄 ATS Score Checker (Auto Mode)")
st.write("Upload your resume (PDF) — the system will automatically calculate your ATS score.")

uploaded_file = st.file_uploader("📁 Upload Resume (PDF format)", type=["pdf"])

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

if uploaded_file is not None:
    if st.button("🔍 Check ATS Score"):
        with st.spinner("Extracting text and analyzing resume..."):
            # Extract text from PDF
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""

            if len(text.strip()) == 0:
                st.error("⚠️ No readable text found in the PDF. Try uploading a text-based resume.")
            else:
                # Clean and vectorize
                cleaned = clean_text(text)
                vec = vectorizer.transform([cleaned])

                # Predict category (optional)
                category = model.predict(vec)[0]

                # ⚙️ Calculate simple ATS score based on keyword density
                total_words = len(cleaned.split())
                score = min(100, int((len(set(cleaned.split())) / total_words) * 100))

                st.success(f"✅ **ATS Score: {score}%**")
                st.info(f"📂 Detected Resume Category: {category}")
