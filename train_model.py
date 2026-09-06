"""
train_model.py

Reproduces the pipeline from Sentiment_Analysis.ipynb as a standalone
script:

    load CSV -> drop NA/duplicates -> preprocess text -> TF-IDF
    -> train Naive Bayes + Linear SVM -> evaluate -> save the SVM model
    (it performed better) and the fitted vectorizer to models/

Run this once locally (or during your Render build step) before starting
the Flask app:

    python train_model.py
"""

import os
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import ensure_nltk_data, preprocess_text

DATA_PATH = os.path.join("data", "sentiment_data.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Same 3 classes used in the original dataset labels
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


def main():
    print("Downloading/verifying NLTK resources...")
    ensure_nltk_data()

    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # ---- Cleaning (same as the notebook) ----
    df = df.dropna(subset=["Comment", "Sentiment"])
    df = df.drop_duplicates(subset=["Comment"])
    print(f"Dataset shape after cleaning: {df.shape}")

    # ---- Text preprocessing ----
    print("Preprocessing text (tokenize, remove stopwords, lemmatize)...")
    start = time.time()
    df["cleaned_comment"] = df["Comment"].apply(preprocess_text)
    print(f"Preprocessing done in {time.time() - start:.1f}s")

    X = df["cleaned_comment"]
    y = df["Sentiment"]

    # ---- Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- TF-IDF vectorization ----
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF training matrix shape: {X_train_tfidf.shape}")

    # ---- Naive Bayes (for comparison, as in the notebook) ----
    print("Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

    # ---- SVM (the model the notebook selected as the winner) ----
    print("Training Linear SVM...")
    svm_model = LinearSVC()
    svm_model.fit(X_train_tfidf, y_train)
    svm_pred = svm_model.predict(X_test_tfidf)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")

    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_pred, target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]))

    # ---- Save the winning model + vectorizer for the Flask app ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svm_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
