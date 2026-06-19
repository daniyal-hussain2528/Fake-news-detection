"""
train_model.py
Trains the Fake News Detection model using the EXACT same pipeline
as the original notebook (fake_news_detection_system.ipynb):

    clean_text() -> TfidfVectorizer(max_features=5000, stop_words='english') -> LinearSVC

Saves:
    model.pkl       -> trained LinearSVC model
    vectorizer.pkl  -> fitted TfidfVectorizer

Run this once locally (or here) to generate the .pkl files,
then upload model.pkl + vectorizer.pkl to your GitHub repo
alongside app.py so the Streamlit app can load them directly
(no retraining needed on deploy).
"""

import re
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ---------- 1. Load data ----------
print("Loading data...")
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0   # 0 = Fake
true["label"] = 1   # 1 = Real

data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42)
data.reset_index(drop=True, inplace=True)


# ---------- 2. Clean text (same function as notebook) ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text


print("Cleaning text...")
data["text"] = data["text"].apply(clean_text)
data.dropna(inplace=True)


# ---------- 3. Features / labels ----------
X = data["text"]
y = data["label"]

print("Vectorizing (TF-IDF, max_features=5000)...")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)


# ---------- 4. Train LinearSVC (the model used in predict_news) ----------
print("Training LinearSVC...")
svm = LinearSVC()
svm.fit(X_train, y_train)

pred = svm.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"\nTest Accuracy: {acc:.4f}\n")
print(classification_report(y_test, pred, target_names=["Fake", "Real"]))


# ---------- 5. Save model + vectorizer ----------
with open("model.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\nSaved model.pkl and vectorizer.pkl successfully.")
