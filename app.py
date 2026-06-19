"""
app.py
Streamlit frontend for the Fake News Detection System.

Loads the pre-trained model.pkl + vectorizer.pkl (trained using the
same pipeline as the original notebook) and lets the user paste
news text to classify it as Real or Fake.

To run locally:
    streamlit run app.py

Required files in the same folder:
    model.pkl
    vectorizer.pkl
    requirements.txt
"""

import re
import pickle

import streamlit as st


# ---------- Page config ----------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)


# ---------- Load model + vectorizer (cached so it loads once) ----------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
    load_error = None
except FileNotFoundError:
    model, vectorizer = None, None
    load_error = (
        "model.pkl or vectorizer.pkl was not found. "
        "Run train_model.py first to generate these files, "
        "then place them in the same folder/repo as app.py."
    )
except Exception as e:
    model, vectorizer = None, None
    load_error = (
        "An error occurred while loading the model (likely a library version "
        "mismatch between the training environment and this server).\n\n"
        f"**Error type:** `{type(e).__name__}`\n\n"
        f"**Error details:** `{e}`"
    )


# ---------- Same clean_text() as the notebook ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text


def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    return "Real News" if pred == 1 else "Fake News"


# ---------- UI ----------
st.title("📰 Fake News Detection System")
st.write(
    "Paste any news article text below, "
    "and the model will tell you whether it's **Real** or **Fake**."
)

if load_error:
    st.error(load_error)
    st.stop()

news_input = st.text_area(
    "Paste the news text here:",
    height=220,
    placeholder="Paste the full news article text here...",
)

col1, col2 = st.columns([1, 4])
with col1:
    check_clicked = st.button("Check News", type="primary", use_container_width=True)
with col2:
    st.caption("Model: TF-IDF + Linear SVM | Trained on Fake.csv / True.csv")

if check_clicked:
    if not news_input.strip():
        st.warning("Please enter or paste some news text first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_news(news_input)

        if result == "Real News":
            st.success(f"✅ Prediction: **{result}**")
        else:
            st.error(f"🚫 Prediction: **{result}**")

st.divider()
st.caption(
    "⚠️ Disclaimer: This model is trained on a specific dataset and is not "
    "100% accurate. Do not rely solely on this tool for important decisions."
)
