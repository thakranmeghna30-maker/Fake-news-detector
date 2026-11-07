import joblib
from pathlib import Path
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords",quiet=True)
stop_words=set(stopwords.words("english"))

def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+ | www\S+","",text)
    text=re.sub(r"[^a-z\s]","",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text
def remove_stopwords(text):
    words=text.split()
    final_text=[word for word in words if word not in stop_words]
    return " ".join(final_text)

BASE_DIR = Path(__file__).resolve().parent
@st.cache_resource
def load_model():
    model=joblib.load(BASE_DIR/"fake_news_pipeline.pkl")
    return model

model=load_model()

# STREAMLIT UI
st.title("Fake News Detector")
user_input=st.text_area("Enter a News Article:")
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a news article first.")
    else:
        cleaned=clean_text(user_input)
        final=remove_stopwords(cleaned)
        prediction=model.predict([final])

        if prediction[0]==0:
            st.error("This news is FAKE")
        else:
            st.success("This news is REAL")

