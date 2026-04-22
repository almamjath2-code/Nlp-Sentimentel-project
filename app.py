import streamlit as st
import joblib
import pandas as pd

from clean_text import clean_text
from clean_lemmitezer import after_cleaning_text

# load model
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# UI
st.title("Sentiment Analysis App 😊")
st.write("Enter your text below:")

# input
user_input = st.text_area("Type here...")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        # Step 1: clean
        cleaned = clean_text(user_input)

        # Step 2: lemmatize
        lemmatized = after_cleaning_text(cleaned)

        # Step 3: vectorize
        X = tfidf.transform([lemmatized])

        # Step 4: predict
        pred = model.predict(X)[0]

        label_map = {
            0: "Negative 😡",
            1: "Neutral 😐",
            2: "Positive 😊"
        }

        # output
        st.success(f"Prediction: {label_map[pred]}")

        # optional UI
        if pred == 2:
            st.balloons()