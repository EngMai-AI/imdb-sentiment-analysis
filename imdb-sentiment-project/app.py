import streamlit as st
import pickle

# Load model + vectorizer
import os

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "Review.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Page config
st.set_page_config(page_title="Sentiment App", page_icon="🎬", layout="centered")

# Header
st.markdown("""
<div style="background-color:#111827;padding:20px;border-radius:15px;text-align:center">
    <h1 style="color:white;">🎬 Movie Review Sentiment Analysis</h1>
    <p style="color:#9CA3AF;">Enter a review and get prediction</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Input
st.markdown("### ✍️ Write your review below:")
user_input = st.text_area("", height=150)

# Button style
st.markdown("""
<style>
.stButton>button {
    background-color:#4F46E5;
    color:white;
    padding:10px 20px;
    border-radius:10px;
    border:none;
    font-size:16px;
}
.stButton>button:hover {
    background-color:#4338CA;
}
</style>
""", unsafe_allow_html=True)

# Predict
if st.button("Analyze Review 🚀"):

    vector = vectorizer.transform([user_input])

    prediction = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]

    negative = probs[0] * 100
    positive = probs[1] * 100

    confidence = max(probs) * 100

    # Result Card
    if prediction == 1:
        st.markdown(f"""
<div style="
    background-color:#0f172a;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:2px solid #22c55e;
">
    <h2 style="color:#22c55e;">😊 Positive Review</h2>
    <h3 style="color:white;">Confidence: {confidence:.2f}%</h3>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="
    background-color:#0f172a;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:2px solid #ef4444;
">
    <h2 style="color:#ef4444;">😞 Negative Review</h2>
    <h3 style="color:white;">Confidence: {confidence:.2f}%</h3>
</div>
""", unsafe_allow_html=True)

    # Breakdown (IMPORTANT 🔥)
    st.write("### 📊 Probability Breakdown")

    st.markdown(f"""
<div style="
    background-color:#111827;
    padding:15px;
    border-radius:12px;
    color:white;
    border:1px solid #374151;
">
    <h4>📊 Probability Breakdown</h4>
    <p>😊 Positive: {positive:.2f}%</p>
    <p>😞 Negative: {negative:.2f}%</p>
</div>
""", unsafe_allow_html=True)

    # Progress bar
    st.write("### Confidence Level")
    st.progress(float(confidence / 100))

# Footer
st.markdown("""
<hr>
<p style="text-align:center;color:gray;">Built with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
