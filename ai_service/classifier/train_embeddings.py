from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from classifier.text import clean

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = next((ROOT / "data").glob("*.csv"))
OUT_PATH = Path(__file__).resolve().parent / "embed_clf.joblib"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

df = pd.read_csv(CSV_PATH)
X = df["instruction"].map(clean).tolist()
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

encoder = SentenceTransformer(ENCODER_NAME)
train_vecs = encoder.encode(X_train, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
test_vecs = encoder.encode(X_test, batch_size=64, normalize_embeddings=True)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(train_vecs, y_train)

pred = clf.predict(test_vecs)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro F1:", round(f1_score(y_test, pred, average="macro"), 4))

joblib.dump(clf, OUT_PATH)
print("Saved classifier head to", OUT_PATH)