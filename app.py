from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="AI DevOps Incident Classifier"
)

model = joblib.load(
    "models/incident_classifier.joblib"
)

class IncidentRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "AI DevOps Incident Classifier Running"
    }

@app.post("/predict")
def predict(data: IncidentRequest):

    prediction = model.predict([data.text])[0]

    probabilities = model.predict_proba([data.text])[0]

    confidence = float(max(probabilities))

    return {
        "incident_type": prediction,
        "confidence": round(confidence, 4)
    }
