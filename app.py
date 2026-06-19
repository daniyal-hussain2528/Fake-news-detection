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
        "model.pkl ya vectorizer.pkl nahi mili. "
        "Pehle train_model.py run karke ye files generate karein, "
        "phir unhe app.py ke saath same folder/repo me rakhein."
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
    "Neeche kisi bhi news article ka text paste karein, "
    "aur model batayega ke ye **Real** hai ya **Fake**."
)

if load_error:
    st.error(load_error)
    st.stop()

news_input = st.text_area(
    "News text yahan paste karein:",
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
        st.warning("Pehle kuch news text likhein ya paste karein.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_news(news_input)

        if result == "Real News":
            st.success(f"✅ Prediction: **{result}**")
        else:
            st.error(f"🚫 Prediction: **{result}**")

st.divider()
st.caption(
    "⚠️ Disclaimer: Ye model ek dataset par train hua hai aur 100% accurate nahi hai. "
    "Important decisions ke liye sirf is tool par bharosa na karein."
)
