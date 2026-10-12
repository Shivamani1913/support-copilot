from fastapi import FastAPI
from classifier.predict import predict_category
from .schemas import ClassificationRequest, ClassificationResponse

CONFIDENCE_THRESHOLD = 0.40 #provesional; we'll tune it on a real test set
app=FastAPI(title="Support Copilot Ai Services")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassificationResponse)
def classify(req:ClassificationRequest):
    #Placeholder for actual classification logic, the trained scikit-learn model replaces this later
    category, confidence = predict_category(req.subject, req.body)
    return ClassificationResponse(
        category=category,
        priority="medium",
        confidence=round(confidence, 4),
        needs_review=confidence<CONFIDENCE_THRESHOLD,
    )