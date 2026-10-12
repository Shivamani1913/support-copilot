from pathlib import Path

import joblib

from classifier.text import clean

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
_model = joblib.load(MODEL_PATH)  # loaded once when the service starts


def predict_category(subject: str, body: str) -> tuple[str, float]:
    text = clean(f"{subject} {body}")
    probs = _model.predict_proba([text])[0]
    best = probs.argmax()
    return str(_model.classes_[best]), float(probs[best])