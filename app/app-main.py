from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import pickle

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class Patient(BaseModel):
    age: float
    bilirubin: float
    albumin: float
    ast: float
    alt: float
    platelets: float
    inr: float


@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.dict()])
    pred = model.predict(df)[0]
    return {"cirrhosis_risk": int(pred)}
