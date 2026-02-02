from pathlib import Path
import re
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# App configuration
st.set_page_config(
    page_title="Flipkart Review Sentiment Analyzer",
    layout="centered"
)

st.title("Flipkart Review Sentiment Analyzer")
st.write("Enter a product review to predict its sentiment.")


# Resolve absolute project paths (NO ambiguity)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(PROJECT_ROOT / "models" / "sentiment_model.pkl")
VECTORIZER_PATH = str(PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl")


# Validate artifacts immediately
if not Path(MODEL_PATH).exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not Path(VECTORIZER_PATH).exists():
    st.error(f"Vectorizer file not found: {VECTORIZER_PATH}")
    st.stop()


# Load artifacts (NO caching for stability)
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# NLTK setup
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(tokens)


# UI
user_review = st.text_area(
    "Enter your review",
    height=150,
    placeholder="Type your review here..."
)

if st.button("Analyze Sentiment"):
    if not user_review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(user_review)
        vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("Positive Review")
        else:
            st.error("Negative Review")
