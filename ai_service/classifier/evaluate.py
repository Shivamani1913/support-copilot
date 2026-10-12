from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from classifier.predict import predict_category

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.40

df = pd.read_csv(ROOT / "evals" / "test_tickets.csv")
results = [predict_category(s, b) for s, b in zip(df["subject"], df["body"])]
df["predicted"] = [r[0] for r in results]
df["confidence"] = [r[1] for r in results]

acc = accuracy_score(df["expected"], df["predicted"])
f1 = f1_score(df["expected"], df["predicted"], average="macro", zero_division=0)
print(f"Tickets: {len(df)}")
print(f"Accuracy: {acc:.3f}   Macro F1: {f1:.3f}")

confident = df[df["confidence"] >= THRESHOLD]
if len(confident):
    conf_acc = accuracy_score(confident["expected"], confident["predicted"])
    print(f"Confident (>= {THRESHOLD}): {len(confident)} tickets, accuracy {conf_acc:.3f}")
print(f"Flagged for human review: {len(df) - len(confident)} tickets")

wrong = df[df["expected"] != df["predicted"]]
print("\nWrong predictions:")
print(wrong[["expected", "predicted", "confidence", "subject"]].to_string(index=False))
