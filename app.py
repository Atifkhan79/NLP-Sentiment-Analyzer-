"""
app.py

Flask app that serves an HTML page where a user can type a product
review and get back a Positive / Neutral / Negative prediction from the
TF-IDF + Linear SVM model trained in train_model.py.

Local run:
    python app.py

Production (Render / gunicorn):
    gunicorn app:app
"""

import os

import joblib
from flask import Flask, jsonify, render_template, request

from preprocessing import preprocess_text

MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
MAX_REVIEW_LENGTH = 2000

app = Flask(__name__)

# Load the trained model + vectorizer once at startup, not per-request.
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    raise FileNotFoundError(
        "Model files not found. Run `python train_model.py` first to "
        "generate models/sentiment_model.pkl and models/tfidf_vectorizer.pkl."
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()

    if not review:
        return jsonify({"error": "Please enter a review before analyzing."}), 400

    if len(review) > MAX_REVIEW_LENGTH:
        review = review[:MAX_REVIEW_LENGTH]

    cleaned = preprocess_text(review)

    if not cleaned:
        return jsonify({
            "review": review,
            "sentiment": "Neutral",
            "note": "No meaningful words were detected after cleaning.",
        })

    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    sentiment = LABEL_MAP.get(int(prediction), str(prediction))

    return jsonify({"review": review, "cleaned": cleaned, "sentiment": sentiment})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
