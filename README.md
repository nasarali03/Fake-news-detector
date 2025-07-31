# 📰 VariNews - Fake News Detector

An AI-powered application to detect fake news using **Natural Language Processing (NLP)** and **Machine Learning**, enhanced with **LLM query normalization**.
Built with **Streamlit**, **scikit-learn**, **spaCy**, **NLTK**, and **LangChain Groq LLMs**.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Confusion Matrix](#confusion-matrix)
- [Streamlit App (VariNews)](#streamlit-app-varinews)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Notes](#notes)

---

## 📌 Project Overview

**VariNews** 📰 is an AI assistant that helps you check whether a news article is **real** ✅ or **fake** ❌.
The workflow includes:

- Data preprocessing (cleaning, stopword removal, lemmatization)
- Feature extraction (TF-IDF)
- Logistic Regression model training
- **LLM query normalization** for better user input handling
- Interactive **Streamlit UI** for real-time detection
- Confidence score visualization with progress bar

---

## 📊 Dataset

- **Source:** [Fake and real news dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files:**

  - `Fake.csv`: Contains fake news articles.
  - `True.csv`: Contains real news articles.

**Columns:**
`title`, `text`, `subject`, `date`

---

## ✨ Features

- Text cleaning and normalization using NLTK and spaCy
- Tokenization & Lemmatization
- TF-IDF vectorization (with n-grams)
- Logistic Regression classifier with class balancing
- LLM-powered query reformulation for better detection
- Confidence score progress bar
- Resettable chat-based UI using Streamlit
- Two modes:

  - 📰 **News Mode** → Detects fake/real news
  - 💬 **Casual Mode** → Friendly greetings & guidance

---

## 📂 Project Structure

```
VariNews/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/
│   ├── main.ipynb
│   ├── sample.ipynb
│
│   ├── app.py                 # Streamlit UI
│   ├── query_normalization.py # Groq LLM integration
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── confusion_matrix.png
│
└── README.md
```

---

## ⚙️ Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd VariNews/app
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Unix/Mac
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLP resources:**

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')

   import spacy
   spacy.cli.download("en_core_web_sm")
   ```

5. **Add your Groq API key:**
   Create a `.env` file in the root directory:

   ```
   GROQ_API_KEY=your_api_key_here
   ```

---

## 🚀 Usage

### 1. Run Streamlit App (Recommended)

```bash
streamlit run app/app.py
```

- Paste a news headline or article.
- See if it's **Real** ✅ or **Fake** ❌.
- Confidence score is displayed with a progress bar.
- Chat resets available via 🔄 button.

### 2. Jupyter Notebook

Run `notebooks/main.ipynb` or `sample.ipynb` for training and evaluation.

---

## 🧠 Model Training & Evaluation

- **Preprocessing:** Lowercasing, removing URLs, HTML tags, punctuation, stopwords, and lemmatization.
- **Feature Extraction:** TF-IDF vectorization with unigrams and bigrams.
- **Model:** Logistic Regression (`class_weight='balanced'`, `max_iter=1000`).
- **Evaluation Metrics:** Accuracy, F1 score, confusion matrix.

---

## 📈 Results

- **Test Accuracy:** \~95.7%
- **F1 Score:** \~0.95
- **Balanced Performance** on both fake and real classes.

---

## 📊 Confusion Matrix

![Confusion Matrix](app/confusion_matrix.png)

---

## 🖥️ Streamlit App (VariNews)

- Interactive UI with **chat-like interface**
- Handles **casual greetings** and redirects user to provide news
- Shows **confidence score bar**
- Expandable panel to view **LLM-reformulated news text**

---

## 📦 Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- spacy
- matplotlib
- seaborn
- streamlit
- langchain
- langchain-groq
- python-dotenv

---

## 📜 License

This project is for educational purposes only.

---

## 🙏 Acknowledgements

- [Kaggle Fake/Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [spaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)

---

## 📝 Notes

If you encounter errors:

- For spaCy model:

  ```python
  import spacy
  spacy.cli.download("en_core_web_sm")
  ```

- For NLTK resources:

  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

---
