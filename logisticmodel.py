import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
df = pd.read_csv("Resume.csv")
print(df.head())

# -----------------------------
# 2️⃣ Download stopwords (only once)
# -----------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# 3️⃣ Clean Text Function
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))  # remove links
    text = re.sub(r'[^a-zA-Z]', ' ', text)                   # remove numbers/special chars
    text = text.lower()                                      # lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Drop NaN rows
df = df.dropna(subset=['Resume_str', 'Category'])

# -----------------------------
# 4️⃣ Apply Cleaning
# -----------------------------
df['Cleaned'] = df['Resume_str'].apply(clean_resume)

# -----------------------------
# 5️⃣ Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned'], df['Category'],
                                                    test_size=0.2, random_state=42)

# -----------------------------
# 6️⃣ TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 7️⃣ Train Model (LOGISTIC REGRESSION)
# -----------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# -----------------------------
# 8️⃣ Evaluate
# -----------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Logistic Regression Model Trained Successfully!")
print(f"🎯 Accuracy: {accuracy:.2f}")

# Save classification report
report = classification_report(y_test, y_pred)
with open("logistic_report.txt", "w") as f:
    f.write(report)

print("📄 Full classification report saved to 'logistic_report.txt'")

# -----------------------------
# 9️⃣ Save Model + Vectorizer
# -----------------------------
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidflogistic_vectorizer.pkl")

print("💾 Logistic Model and Vectorizer saved successfully!")
