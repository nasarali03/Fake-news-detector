import streamlit as st
import pickle
import re
import spacy
import string
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from query_normalization import query_groq_llm, query_groq_llm_casual
import numpy as np

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to detect casual conversation
def is_casual_conversation(text: str) -> bool:
    text_lower = text.lower().strip()
    greeting_patterns = [r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b']
    thank_patterns = [r'\b(thanks?|thank you|thx|ty)\b']
    casual_patterns = [r'\b(bye|goodbye|see you|take care)\b']
    all_patterns = greeting_patterns + thank_patterns + casual_patterns
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            return True
    if len(text.split()) <= 3 and len(text) < 50:
        return True
    return False

# Load models
@st.cache_resource
def load_models():
    with open('./model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('./model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    nlp = spacy.load('en_core_web_sm')
    return vectorizer, model, nlp

vectorizer, model, nlp = load_models()

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])
    return text

# UI Config
st.set_page_config(page_title="VariNews 📰", page_icon="📰", layout="centered")

# Header Branding
st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#2E86C1;">📰 VariNews</h1>
        <p style="font-size:18px;">Your AI-powered Fake News Detective</p>
        <hr>
    </div>
""", unsafe_allow_html=True)

# Reset button
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "prediction" in message:
            st.write(f"🤖 Prediction: {message['prediction']}")
            if "confidence" in message:
                st.progress(int(message['confidence'] * 100))
                st.caption(f"Confidence: {message['confidence']:.2%}")

# Input box
user_input = st.chat_input("Type or paste your news headline or article...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    if is_casual_conversation(user_input):
        with st.chat_message("assistant"):
            st.info("💬 Processing your casual message...")
            time.sleep(1)
            llm_response = query_groq_llm_casual(user_input)
            st.write(llm_response)

        st.session_state.chat_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": llm_response}
        ])

    else:
        llm_response = query_groq_llm(user_input, conversation_type="news")
        with st.expander("🔍 See how VariNews reformulated your query"):
            st.write(llm_response)

        processed = preprocess_text(llm_response)
        tokens = word_tokenize(processed)
        text_for_vectorizer = ' '.join(tokens)
        X = vectorizer.transform([text_for_vectorizer])

        proba = model.predict_proba(X)[0]
        pred = np.argmax(proba)
        label = '✅ Real News' if pred == 1 else '❌ Fake News'
        confidence = proba[pred]

        with st.chat_message("assistant"):
            st.write("Analyzing the news text... ⏳")
            time.sleep(1.5)
            st.success(f"Prediction: {label}")
            st.progress(int(confidence * 100))
            st.caption(f"Confidence Level: {confidence:.2%}")

        st.session_state.chat_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", 
             "content": f"Prediction: {label}", 
             "prediction": label, 
             "confidence": confidence}
        ])

# Footer Branding
st.markdown("""
    <hr>
    <div style="text-align:center; font-size:14px; color:gray;">
        🚀 Powered by VariNews | Built with Streamlit & AI 🧠
    </div>
""", unsafe_allow_html=True)
