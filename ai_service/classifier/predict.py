from pathlib import Path

import joblib

from classifier.text import clean

HERE = Path(__file__).resolve().parent
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_tfidf = None
_encoder = None
_embed_clf = None


def _load_tfidf():
    global _tfidf
    if _tfidf is None:
        _tfidf = joblib.load(HERE / "model.joblib")
    return _tfidf


def _load_embed():
    global _encoder, _embed_clf
    if _embed_clf is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(ENCODER_NAME)
        _embed_clf = joblib.load(HERE / "embed_clf.joblib")
    return _encoder, _embed_clf


def predict_category(subject: str, body: str, backend: str = "tfidf") -> tuple[str, float]:
    text = clean(f"{subject} {body}")
    if backend == "embed":
        encoder, clf = _load_embed()
        probs = clf.predict_proba(encoder.encode([text], normalize_embeddings=True))[0]
        classes = clf.classes_
    else:
        model = _load_tfidf()
        probs = model.predict_proba([text])[0]
        classes = model.classes_
    best = probs.argmax()
    return str(classes[best]), float(probs[best])