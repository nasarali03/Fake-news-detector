import pickle
import re
import spacy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load spacy model
nlp = spacy.load('en_core_web_sm')

# Preprocessing function (same as training)
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

if __name__ == "__main__":
    user_input = input("Enter news text to check if it's fake or real: ")
    processed = preprocess_text(user_input)
    tokens = word_tokenize(processed)
    text_for_vectorizer = ' '.join(tokens)
    X = vectorizer.transform([text_for_vectorizer])
    pred = model.predict(X)[0]
    label = 'Real' if pred == 1 else 'Fake'
    print(f"Prediction: {label}")
