# Review Sentiment Analyzer

A machine learning web app that reads an unstructured product review and
classifies it as **Negative**, **Neutral**, or **Positive**.

Built on a classic NLP pipeline — tokenization, stopword removal,
lemmatization, TF‑IDF vectorization, and a Linear SVM classifier — wrapped
in a Flask API with a simple browser UI, ready to deploy on
[Render](https://render.com).

**Live pipeline:**

```
Raw review text
      ↓
Lowercase + strip punctuation
      ↓
Tokenize (NLTK)
      ↓
Remove stopwords
      ↓
Lemmatize
      ↓
TF-IDF vectorization (scikit-learn)
      ↓
Linear SVM classifier
      ↓
Negative / Neutral / Positive
```

## Results

Trained and evaluated on 214k+ cleaned, de-duplicated comments
(`data/sentiment_data.csv`), an 80/20 stratified train/test split:

| Model              | Accuracy |
|---------------------|----------|
| Multinomial Naive Bayes | 64.5% |
| **Linear SVM (used in app)** | **79.6%** |

```
              precision    recall  f1-score   support

    Negative       0.78      0.68      0.73      9654
     Neutral       0.76      0.83      0.79     14339
    Positive       0.84      0.83      0.83     18839

    accuracy                           0.80     42832
```

The original exploration notebook is in [`notebooks/Sentiment_Analysis.ipynb`](notebooks/Sentiment_Analysis.ipynb).

## Project structure

```
.
├── app.py                  # Flask app (routes + inference)
├── train_model.py          # Reproducible training script
├── preprocessing.py        # Shared text-cleaning pipeline (train + serve)
├── data/
│   └── sentiment_data.csv  # Training data (Comment, Sentiment columns)
├── models/
│   ├── sentiment_model.pkl       # Trained Linear SVM (pre-built, committed)
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer (pre-built, committed)
├── notebooks/
│   └── Sentiment_Analysis.ipynb  # Original EDA / experimentation notebook
├── templates/
│   └── index.html          # Single-page UI
├── static/
│   ├── style.css
│   └── script.js
├── requirements.txt
├── Procfile                # gunicorn start command
├── render.yaml              # Render blueprint (infra-as-code)
└── .gitignore
```

## Run it locally

```bash
git clone https://github.com/<your-username>/sentiment-analysis-flask.git
cd sentiment-analysis-flask

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Model + vectorizer are already committed in models/, so you can skip
# straight to running the app. To retrain from scratch instead:
python train_model.py

python app.py
```

Open `http://localhost:5000`.

## Deploy on Render

1. Push this repo to GitHub.
2. In Render, choose **New → Web Service** and connect the repo (or use
   **New → Blueprint** to pick up `render.yaml` automatically).
3. Settings if configuring manually:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
   - **Environment:** Python 3
4. Deploy. The committed `models/*.pkl` files mean Render doesn't need to
   retrain on every build — it just loads them and serves predictions.

> Retraining on Render's build step is intentionally **not** wired up by
> default, since a 200k-row TF‑IDF + SVM fit is slow on a free-tier build
> and isn't needed once the model files are committed. If you update
> `data/sentiment_data.csv` and want a fresh model, run
> `python train_model.py` locally and commit the new `.pkl` files.

## API

`POST /predict`

```json
{ "review": "Shipping was fast but the case cracked on the first drop." }
```

Response:

```json
{
  "review": "Shipping was fast but the case cracked on the first drop.",
  "cleaned": "shipping fast case cracked first drop",
  "sentiment": "Negative"
}
```

## Tech stack

- **NLP:** NLTK (tokenization, stopwords, WordNet lemmatizer)
- **Vectorization / ML:** scikit-learn (`TfidfVectorizer`, `LinearSVC`, `MultinomialNB`)
- **Backend:** Flask, gunicorn
- **Frontend:** vanilla HTML/CSS/JS
- **Deployment:** Render
