import streamlit as st
import pickle

# load model + vectorizer
model = pickle.load(open('Review.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("🎬 Movie Review Sentiment Analysis")

user_input = st.text_area("Write your review here:")

if st.button("Predict"):

    vector = vectorizer.transform([user_input])

    prediction = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    confidence = max(probs)

    label = "Positive 😄" if prediction == 1 else "Negative 😞"

    if prediction == 1:
        st.success(f"{label} ({confidence*100:.2f}%)")
    else:
        st.error(f"{label} ({confidence*100:.2f}%)")