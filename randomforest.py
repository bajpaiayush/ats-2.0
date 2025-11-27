import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Resume.csv")
print(df.head())

# -----------------------------
# Download stopwords
# -----------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# Clean Text Function
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))   # remove links
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))               # remove numbers/special chars
    text = text.lower()                                       # lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Drop NaN rows
df = df.dropna(subset=['Resume_str', 'Category'])

# -----------------------------
#  Apply Cleaning
# -----------------------------
df['Cleaned'] = df['Resume_str'].apply(clean_resume)

# -----------------------------
#  Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['Cleaned'], df['Category'], test_size=0.2, random_state=42
)

# -----------------------------
#  TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
#  Train Model (RANDOM FOREST)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,       # number of trees
    max_depth=None,         # let trees grow fully
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
model.fit(X_train_vec, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Random Forest Model Trained Successfully!")
print(f"🎯 Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
with open("random_forest_report.txt", "w") as f:
    f.write(report)

print("📄 Full classification report saved to 'random_forest_report.txt'")

# -----------------------------
#  Save Model + Vectorizer
# -----------------------------
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(vectorizer, "rftfidf_vectorizer.pkl")

print("💾 Random Forest Model and Vectorizer saved successfully!")
