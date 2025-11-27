import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
df = pd.read_csv("Resume.csv")
print(df.head())

# -----------------------------
# 2️⃣ Download stopwords
# -----------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# 3️⃣ Clean Text
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df = df.dropna(subset=['Resume_str', 'Category'])

# -----------------------------
# 4️⃣ Apply Cleaning
# -----------------------------
df['Cleaned'] = df['Resume_str'].apply(clean_resume)

# -----------------------------
# 5️⃣ Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned'], df['Category'],
                                                    test_size=0.2, random_state=42)

# -----------------------------
# 6️⃣ TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 7️⃣ Train Model (Naive Bayes)
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# 8️⃣ Evaluate
# -----------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Naive Bayes Model Trained Successfully!")
print(f"🎯 Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
with open("naive_bayes_report.txt", "w") as f:
    f.write(report)

print("📄 Full classification report saved to 'naive_bayes_report.txt'")

# -----------------------------
# 9️⃣ Save Model Files
# -----------------------------
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidfnaivebayes_vectorizer.pkl")

print("💾 Naive Bayes Model and Vectorizer saved successfully!")
