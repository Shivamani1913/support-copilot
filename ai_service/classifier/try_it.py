from pathlib import Path
import joblib
model=joblib.load(Path(__file__).resolve().parent/"model.joblib")

tickets = [
    "my package never arrived and tracking hasn't moved in a week",
    "charged twice for the same thing, want my money back",
    "cant log in, password reset email never comes",
    "how do i change my delivery address",
    "your app is great but the checkout page is slow",
    "please cancel my subscription before it renews",
    "i need a copy of my invoice for last month",
    "wanna talk to a real person",
    "ordered the wrong size, how do i return it",
    "my card got declined at checkout",
]

probs=model.predict_proba(tickets)
labels=model.classes_
for ticket, p in zip(tickets, probs):
    best=p.argmax()
    print(f"{labels[best]:<13}{p[best]: .2f}  {ticket}")
