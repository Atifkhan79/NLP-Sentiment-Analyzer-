# 🧠 NLP Sentiment Analysis

A Machine Learning project that automatically classifies human-written text, such as product reviews, into **Positive** or **Negative** sentiment using Natural Language Processing (NLP), TF-IDF vectorization, and Machine Learning classifiers.

---

## 📌 Project Overview

Explore this Project =>

Sentiment Analysis is an NLP task that determines the emotional tone of text.

In this project, a machine learning model is trained to understand customer reviews and classify them as:

* 🟢 **Positive**
* 🔴 **Negative**

The project implements a complete NLP pipeline, starting from raw unstructured text and transforming it into numerical features that can be processed by machine learning algorithms.

### Workflow

```text
Raw Reviews
     ↓
Data Cleaning
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Stop-Word Removal
     ↓
Lemmatization
     ↓
TF-IDF Vectorization
     ↓
Machine Learning
     ↓
Model Evaluation
     ↓
Positive / Negative Prediction
```

---

## 🎯 Objectives

The main objectives of this project are:

* Process unstructured human text.
* Build a complete NLP preprocessing pipeline.
* Perform tokenization.
* Remove unnecessary stop words.
* Apply lemmatization.
* Convert text into numerical vectors using TF-IDF.
* Train Machine Learning classification models.
* Compare Naive Bayes and SVM performance.
* Evaluate the models using accuracy, precision, recall, and F1-score.
* Predict the sentiment of new reviews.

---

## 🛠️ Technologies & Libraries

### Programming Language

* 🐍 Python

### NLP

* NLTK
* Regular Expressions

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Naive Bayes
* Support Vector Machine (SVM)

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook

---

## 📂 Project Structure

```text
NLP-Sentiment-Analysis/
│
├── data/
│   └── dataset.csv
│
├── NLP_Sentiment_Analysis.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 📊 Dataset

The dataset contains text reviews and their corresponding sentiment labels.

Example:

| Review                           | Sentiment |
| -------------------------------- | --------- |
| "This product is amazing!"       | Positive  |
| "I love this product."           | Positive  |
| "Excellent quality and service." | Positive  |
| "This product is terrible."      | Negative  |
| "Very poor quality."             | Negative  |
| "I hate this product."           | Negative  |

### Dataset Columns

```text
review
sentiment
```

Where:

* `review` → Customer's written review.
* `sentiment` → Target label: `positive` or `negative`.

---

# 🔄 NLP Preprocessing Pipeline

Raw human language contains punctuation, unnecessary words, different word forms, and other noise.

Therefore, preprocessing is performed before training the model.

## 1. Lowercase Conversion

Example:

```text
"I LOVE This Product"
```

becomes:

```text
"i love this product"
```

This prevents words with different capitalization from being treated as different features.

---

## 2. Remove Punctuation

Example:

```text
"This product is amazing!!!"
```

becomes:

```text
"This product is amazing"
```

Regular expressions are used to remove unnecessary characters.

---

## 3. Tokenization

The sentence is divided into individual words.

```text
"This product is amazing"
```

becomes:

```python
["this", "product", "is", "amazing"]
```

---

## 4. Stop-Word Removal

Common words that provide limited information for classification are removed.

Examples:

```text
the
is
a
an
this
and
of
to
```

Example:

```text
"this product is amazing"
```

becomes approximately:

```text
["product", "amazing"]
```

---

## 5. Lemmatization

Lemmatization converts words into their meaningful base forms.

Examples:

```text
cars → car
studies → study
running → run
```

This helps reduce unnecessary variations of words.

---

# 🔢 TF-IDF Vectorization

Machine learning algorithms cannot directly understand text.

Therefore, cleaned text is converted into numerical vectors using:

**TF-IDF — Term Frequency-Inverse Document Frequency**

```text
Clean Text
     ↓
TF-IDF Vectorizer
     ↓
Numerical Feature Vectors
     ↓
Machine Learning Model
```

TF-IDF assigns importance to words based on how frequently they occur in a document and how common or rare they are across the dataset.

Example:

```text
"I love this product"
```

is transformed into a numerical representation such as:

```text
[0.00, 0.42, 0.18, 0.65, ...]
```

These numerical features can then be used by the machine learning classifier.

---

# 🤖 Machine Learning Models

Two classification algorithms are implemented and compared.

## 1. Multinomial Naive Bayes

Naive Bayes is a probabilistic classification algorithm that works particularly well for many text-classification tasks.

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)
```

---

## 2. Support Vector Machine

A Linear Support Vector Machine is also trained for sentiment classification.

```python
from sklearn.svm import LinearSVC

model = LinearSVC()

model.fit(X_train_tfidf, y_train)
```

SVM is commonly effective for high-dimensional text features such as TF-IDF vectors.

---

# 📈 Model Evaluation

The trained models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Accuracy

Measures the percentage of predictions that are correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive/negative examples were actually correct.

### Recall

Measures how many actual examples of a class were correctly identified.

### F1-Score

Provides a balance between precision and recall.

---

# 📊 Results

After training, the performance of both models can be compared.

| Model       |        Accuracy |
| ----------- | --------------: |
| Naive Bayes | Add Your Result |
| SVM         | Add Your Result |

> **Note:** Replace the values above with the actual accuracy obtained when running the notebook.

Example:

```text
Naive Bayes Accuracy: 0.XX
SVM Accuracy:         0.XX
```

The model with the best validation/test performance can be selected as the final sentiment classifier.

---

# 🧪 Example Predictions

### Positive Review

**Input:**

```text
I absolutely love this product. It is amazing!
```

**Prediction:**

```text
Positive
```

### Negative Review

**Input:**

```text
This product is terrible and completely useless.
```

**Prediction:**

```text
Negative
```

---

# 🧩 Prediction Pipeline

For a new review, the same preprocessing and vectorization steps are applied:

```text
New Review
    ↓
Lowercase
    ↓
Remove Punctuation
    ↓
Tokenization
    ↓
Stop-Word Removal
    ↓
Lemmatization
    ↓
TF-IDF Transformation
    ↓
Trained SVM Model
    ↓
Prediction
    ↓
Positive / Negative
```

Example:

```python
new_review = "This product is absolutely amazing"

clean_review = preprocess_text(new_review)

review_vector = vectorizer.transform(
    [clean_review]
)

prediction = svm_model.predict(
    review_vector
)

print("Prediction:", prediction[0])
```

Output:

```text
Prediction: positive
```

---

# 💡 Key Concepts Learned

Through this project, the following concepts were implemented:

### Natural Language Processing

* Text preprocessing
* Tokenization
* Stop-word removal
* Lemmatization
* Unstructured text processing

### Feature Engineering

* TF-IDF
* Text vectorization
* Numerical representation of language

### Machine Learning

* Supervised learning
* Binary classification
* Naive Bayes
* Support Vector Machine

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

# 🚀 Future Improvements

Possible improvements include:

* [ ] Add a web interface using Streamlit.
* [ ] Add Neutral sentiment classification.
* [ ] Experiment with Word2Vec embeddings.
* [ ] Experiment with GloVe embeddings.
* [ ] Implement Logistic Regression.
* [ ] Try Random Forest.
* [ ] Use spaCy for NLP preprocessing.
* [ ] Perform hyperparameter tuning.
* [ ] Handle emojis and internet slang.
* [ ] Handle negation such as `"not good"`.
* [ ] Deploy the model as an API.
* [ ] Deploy the application online.
* [ ] Add real-time review prediction.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/NLP-Sentiment-Analysis.git
```

Navigate into the project:

```bash
cd NLP-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
NLP_Sentiment_Analysis.ipynb
```

and run the cells sequentially.

---

# 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
jupyter
```

---

# 👨‍💻 Author

**Atif Khan**

### Skills

* Python
* Machine Learning
* Natural Language Processing
* Artificial Intelligence
* Full Stack Development
* MERN Stack
* Generative AI
* LLM Applications
* RAG & AI Automation

---

# ⭐ If You Find This Project Useful

If you find this project helpful, consider giving the repository a ⭐ on GitHub.

This project is intended for educational and portfolio purposes.
