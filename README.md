# Fake News Detector

A machine learning project to detect fake news using Natural Language Processing (NLP) techniques. This project uses a Logistic Regression model trained on a dataset of real and fake news articles, with preprocessing and feature extraction using NLTK and spaCy.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Confusion Matrix](#confusion-matrix)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Notes](#notes)

---

## Project Overview

This project aims to classify news articles as **fake** or **real** using machine learning. The workflow includes:

- Data loading and labeling
- Text preprocessing (cleaning, stopword removal, lemmatization)
- Feature extraction (TF-IDF)
- Model training (Logistic Regression)
- Evaluation (accuracy, F1, confusion matrix)

---

## Dataset

- **Source:** [Fake and real news dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files:**
  - `Fake.csv`: Contains fake news articles.
  - `True.csv`: Contains real news articles.

Each file includes columns: `title`, `text`, `subject`, `date`.

---

## Features

- Text cleaning and normalization using NLTK and spaCy
- Tokenization and lemmatization
- TF-IDF vectorization
- Logistic Regression classifier
- Model evaluation (accuracy, F1, confusion matrix)
- Model and vectorizer serialization (`model.pkl`, `vectorizer.pkl`)

---

## Project Structure

```
skil bridge/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
└── Fake-new-detector/
    ├── fake_news_detector.py
    ├── main.ipynb
    ├── model.pkl
    ├── sample.ipynb
    ├── testing.py
    ├── vectorizer.pkl
    └── confusion_matrix.png   # <-- Save your confusion matrix image here
```

---

**How to save the confusion matrix image:**

In your notebook, after plotting the confusion matrix, add:

```python
plt.savefig('Fake-new-detector/confusion_matrix.png')
```

Then commit the image to your repository.

Let me know if you need a `requirements.txt` or any other help!

---

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd skil\ bridge/Fake-new-detector
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not present, install manually:

   ```bash
   pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn
   ```

4. **Download NLTK and spaCy resources:**

   ```python
   # In a Python shell or notebook:
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')

   import spacy
   spacy.cli.download("en_core_web_sm")
   ```

---

## Usage

### 1. **Jupyter Notebook**

Open and run `main.ipynb` or `sample.ipynb` for step-by-step code, explanations, and results.

### 2. **Python Script**

You can use `fake_news_detector.py` or `testing.py` to run the model on new data or for batch predictions.

---

## Model Training & Evaluation

- **Preprocessing:** Lowercasing, removing URLs, HTML tags, punctuation, stopwords, and lemmatization.
- **Feature Extraction:** TF-IDF vectorization of processed text.
- **Model:** Logistic Regression with balanced class weights.
- **Evaluation:** Accuracy, F1 score, R2 score, MSE, and confusion matrix.

---

## Results

- **Test Accuracy:** ~95.7%
- **F1 Score:** ~0.95

---

## Confusion Matrix

Below is the confusion matrix for the model's predictions on the test set:

![Confusion Matrix](Fake-new-detector/confusion_matrix.png)

---

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- spacy
- matplotlib
- seaborn

---

## License

This project is for educational purposes.

---

## Acknowledgements

- [Fake and real news dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [spaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)

---

## Notes

- If you encounter `OSError: [E050] Can't find model 'en_core_web_sm'`, run:
  ```python
  import spacy
  spacy.cli.download("en_core_web_sm")
  ```
- For NLTK stopwords/tokenizer errors, run:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

---
