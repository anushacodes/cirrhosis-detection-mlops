# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import pickle

# with open("models/model.pkl", "rb") as f:
#         model = pickle.load(f)  

# app = FastAPI()

# class Patient(BaseModel):
#     age: float
#     bilirubin: float
#     albumin: float
#     ast: float
#     alt: float
#     platelets: float
#     inr: float

# @app.post("/predict")
# def predict(patient: Patient):
#     df = pd.DataFrame([patient.model_dump()])
#     pred = model.predict(df)[0]

#     return {"cirrhosis_risk": int(pred)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.responses import JSONResponse
import uvicorn
import joblib
# from mangum import Mangum
import xgboost as xgb


model = xgb.XGBClassifier()
model.load_model("models/xgb_model.json")

# model = joblib.load("models/model.pkl")

# with open("models/model.pkl", "rb") as file:
#     model = pickle.load(file)

app=FastAPI()
# handler=Mangum(app)

class Patient(BaseModel):
    age: float
    bilirubin: float
    albumin: float
    ast: float
    alt: float
    platelets: float
    inr: float

# prediction endpoint
@app.post("/predict")

def predict_diabetes(input_data: Patient):

    data = np.array([[input_data.age, input_data.bilirubin, input_data.albumin,
                      input_data.ast, input_data.alt, input_data.platelets,
                      input_data.inr]])

    prediction = model.predict(data)
    result = "Diabetic" if prediction[0] == 1 else "Non-diabetic"

    return JSONResponse({"prediction": result})


if __name__=="__main__":
  uvicorn.run(app,host="0.0.0.0",port=8080)