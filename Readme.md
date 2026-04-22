# 📘 NLP Sentiment Analysis Project

This project is a complete **Natural Language Processing (NLP) pipeline** for sentiment analysis using Python and machine learning.

---

# 🚀 Project Overview

The goal of this project is to:

* Clean and preprocess text data(including stopword removal)
* Perform Exploratory Data Analysis (EDA)
* Convert text into numerical features using TF-IDF
* Train a machine learning model (LinearSVC)
* Evaluate performance using classification metrics

---

# 📦 Libraries Used

```python
import nltk
import pandas as pd
import contractions
import re
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
```

---

# 📥 NLTK Downloads
## ⚠️ Important Notes

* ❌ Do NOT include the following in `requirements.txt`:

  * `re` (Python built-in module)
  * `collections` (Python built-in module)

* ❌ `nltk.download()` should NOT be added to `requirements.txt`

* ✅ After installing requirements, run the following separately to download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```


Run this once before starting:

```python
nltk.download('stopwords')
nltk.download('wordnet')
```

---

# 🧹 Text Preprocessing Steps

* Lowercasing text
* Removing HTML tags
* Expanding contractions (don’t → do not)
* Removing special characters
* Removing stopwords
* Lemmatization using WordNetLemmatizer

---

# 📊 Exploratory Data Analysis (EDA)

* Class distribution (positive / negative / neutral)
* Word frequency analysis
* Word cloud ☁️

---

# 🧠 Feature Extraction

We use **TF-IDF Vectorization**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

---

# 🤖 Model Training

We use **Linear Support Vector Classifier (LinearSVC)**

Steps:

* Split train/test data
* Train model on TF-IDF features
* Predict sentiment

---

# 📈 Model Evaluation Results

## 🔹 TRAIN SET

* Accuracy : **0.7394 (73.94%)**
* Precision: **0.7414**
* Recall   : **0.7110**
* F1-score : **0.7197**

## 🔹 TEST SET

* Accuracy : **0.7114 (71.14%)**
* Precision: **0.7081**
* Recall   : **0.6802**
* F1-score : **0.6882**

## 🔍 Model Fit Check

✔ Good Fit (No overfitting, good generalization)

---

# 📊 Confusion Matrix (Summary)

Model performs well across all classes (Positive / Negative / Neutral) with balanced predictions.

---

# 💾 Model Saving

```python
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

---

# 🌐 App.py (Flask / Streamlit App)

This project includes an **app.py** file to run predictions in real time.

## ▶️ Run App

```bash
python app.py
```

## 📌 What app.py does:

* Loads trained model
* Loads TF-IDF vectorizer
* Takes user input text
* Cleans text
* Predicts sentiment
* Shows output (Positive / Negative / Neutral)

---

# ⚙️ Requirements

Install dependencies:

```bash
pip install pandas nltk contractions scikit-learn matplotlib wordcloud joblib
```

---

# ▶️ How to Run Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run model training (if needed)

```bash

```

### 3️⃣ Run application

```bash
streamlit run app.py
```

---

# 📊 Output Example

* Positive 😊
* Negative 😡
* Neutral 😐

---

# 🚀 Future Improvements

* Use BERT / LSTM for better accuracy
* Improve preprocessing (emoji + slang handling)
* Add web UI (Streamlit dashboard)
* Hyperparameter tuning

---

# 👨‍💻 Author

Built for learning NLP, machine learning pipeline, and sentiment
