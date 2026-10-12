import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = next((ROOT / "data").glob("*.csv"))
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def clean(text: str) -> str:
    text = re.sub(r"\{\{.*?\}\}", " ", str(text))  # drop template placeholders
    return re.sub(r"\s+", " ", text).strip().lower()


df = pd.read_csv(CSV_PATH)
X = df["instruction"].map(clean)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro F1:", round(f1_score(y_test, pred, average="macro"), 4))
print()
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print("Saved model to", MODEL_PATH)