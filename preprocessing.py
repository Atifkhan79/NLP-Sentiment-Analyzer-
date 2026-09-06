"""
preprocessing.py

The exact NLP cleaning pipeline used in the original Jupyter notebook
(Sentiment_Analysis.ipynb):

    lowercase -> remove punctuation/numbers -> tokenize
    -> remove stopwords -> lemmatize -> rejoin

This module is imported by BOTH train_model.py (to build the training
data) and app.py (to clean incoming reviews at prediction time), so the
exact same transformation is applied every time.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_NLTK_READY = False
_stop_words = None
_lemmatizer = None


def ensure_nltk_data():
    """Download the NLTK resources needed for preprocessing (idempotent)."""
    global _NLTK_READY, _stop_words, _lemmatizer

    if _NLTK_READY:
        return

    for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            # punkt_tab doesn't exist on older nltk versions - safe to ignore
            pass

    _stop_words = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _NLTK_READY = True


def preprocess_text(text: str) -> str:
    """Clean a single raw text string down to lemmatized, stopword-free tokens."""
    ensure_nltk_data()

    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. Tokenization
    tokens = nltk.word_tokenize(text)

    # 4. Stop-word removal
    tokens = [word for word in tokens if word not in _stop_words]

    # 5. Lemmatization
    tokens = [_lemmatizer.lemmatize(word) for word in tokens]

    # 6. Rejoin into a cleaned sentence
    return " ".join(tokens)
