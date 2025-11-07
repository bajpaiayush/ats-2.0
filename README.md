# 🧠 ATS Resume Classifier — Streamlit App

## Overview
This project is an **AI-powered Resume Classifier** that categorizes resumes based on their textual content. It uses **TF-IDF vectorization** and a **Support Vector Machine (SVM)** classifier to predict resume categories. The app is deployed using **Streamlit** for interactive usage.

## 🚀 Features
- Upload and process resume data from CSV
- Clean and preprocess resume text using NLTK
- Train/test split with TF-IDF vectorization
- SVM-based classification
- Performance metrics: Accuracy, Precision, Recall, F1 Score
- Graphical visualization of model performance
- Streamlit-based UI for real-time predictions

## 📊 Model Performance
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.64    |
| Precision  | 0.6572  |
| Recall     | 0.6358  |
| F1 Score   | 0.6355  |

> Metrics are computed using weighted averages to handle class imbalance.

## 📈 Visualizations
- Bar chart of Accuracy, Precision, Recall, F1 Score using `matplotlib`
- Classification report saved as `classification_report.txt`
  <img width="800" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/5e20d1f6-e9d6-4998-b4cd-dc15ef3b2536" />


## 🧪 Tech Stack
- Python
- Pandas, NLTK, Scikit-learn
- Matplotlib
- Streamlit
- Joblib (for model persistence)

## 📂 File Structure
├── modeltrain.py # Model training and evaluation script

├── Resume.csv # Input dataset

├── svm_model.pkl # Saved SVM model

├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer

├── classification_report.txt

├── app.py # Streamlit deployment script
