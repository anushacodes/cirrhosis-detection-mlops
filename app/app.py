from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi.responses import JSONResponse
import uvicorn
import joblib
# from mangum import Mangum
import xgboost as xgb
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


model = xgb.XGBClassifier()
model.load_model("models/model.json")
preprocessor = joblib.load("models/processing.pkl")


app = FastAPI()
# handler=Mangum(app)

class Patient(BaseModel):
    Sex: str = "F"
    Drug: str = "Placebo"
    Ascites: str = "N"
    Hepatomegaly: str = "N"
    Spiders: str = "N"
    Edema: str = "N"
    Status: str = "C"
    Age: float
    Bilirubin: float
    Albumin: float
    Alk_Phos: float
    SGOT: float
    Platelets: float
    Prothrombin: float


@app.get("/")
def read_root():
    return {"message": "running OK."}


@app.post("/predict")
def predict(input_data: Patient):

    df = pd.DataFrame([input_data.model_dump()])
    df_processed = preprocessor(df)

    preds = model.predict(df_processed)
    result = int(preds[0])

    label_map = {0: "No Cirrhosis", 1: "Cirrhosis", 2: "Severe Cirrhosis"}
    label = label_map.get(result, "Unknown")

    return JSONResponse({"prediction": label})



if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8080)