from fastapi import FastAPI
app=FastAPI(title="Support Copilot Ai Services")
@app.get("/health")
def health():
    return {"status": "ok"}