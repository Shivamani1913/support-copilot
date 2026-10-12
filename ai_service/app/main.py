from fastapi import FastAPI
from .schemas import ClassificationRequest, ClassificationResponse
app=FastAPI(title="Support Copilot Ai Services")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassificationResponse)
def classify(req:ClassificationRequest):
    #Placeholder for actual classification logic, the trained scikit-learn model replaces this later
    return ClassificationResponse(category="unclassified", priority="medium", confidence=0.0)