import streamlit as st
import joblib
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean text
def clean_resume(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Streamlit UI
st.title("📄 Resume Category Predictor")

uploaded_file = st.file_uploader("Upload your Resume (PDF format)", type=["pdf"])

if uploaded_file is not None:
    if st.button("Predict"):
        st.write("Extracting text...")
        text = extract_text_from_pdf(uploaded_file)
        if text.strip() == "":
            st.error("No text found in PDF!")
        else:
            cleaned_text = clean_resume(text)
            vectorized = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized)[0]
            st.success(f"Predicted Resume Category: **{prediction}**")
