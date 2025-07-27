import streamlit as st
import pickle
import re
import spacy
import string
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load saved models and spaCy NLP pipeline
@st.cache_resource
def load_models():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    nlp = spacy.load('en_core_web_sm')
    return vectorizer, model, nlp

vectorizer, model, nlp = load_models()

# Text preprocessing function
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

# Streamlit UI
st.title("🕵️‍♂️ Fake News Detective")
st.write("Enter a news article to find out if it's **Real** or **Fake**!")

# Show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "prediction" in message:
            st.write(f"🤖 Prediction: {message['prediction']}")

# Chat input box
user_input = st.chat_input("Type or paste your news article here...")

if user_input:
    # Show user's message
    with st.chat_message("user"):
        st.write(user_input)

    # Preprocess and predict
    processed = preprocess_text(user_input)
    tokens = word_tokenize(processed)
    text_for_vectorizer = ' '.join(tokens)
    X = vectorizer.transform([text_for_vectorizer])
    pred = model.predict(X)[0]
    label = 'Real ✅' if pred == 1 else 'Fake ❌'

    # Show assistant's response with delay
    with st.chat_message("assistant"):
        st.write("Analyzing the news text... ⏳")
        time.sleep(2)  # Delay to simulate processing
        st.write("I've analyzed the news text.")
        st.write(f"Prediction: {label}")

    # Update chat history
    st.session_state.chat_history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": "I've analyzed the news text.", "prediction": label}
    ])
