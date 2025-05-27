from fastapi import FastAPI
import mlflow, pandas as pd, os
from pydantic import BaseModel
from typing import List

app = FastAPI()
_ml_model = None

def load_model():
    global _ml_model
    _ml_model = mlflow.pyfunc.load_model("models:/seoul_temp/Production")

class WeatherRow(BaseModel):
    nx:int; ny:int; hour:int; temp:float; rain:float

@app.on_event("startup")
def startup():
    load_model()

@app.post("/predict")
def predict(rows: List[WeatherRow]):
    df = pd.DataFrame([r.dict() for r in rows])
    preds = _ml_model.predict(df)
    return {"prediction": preds.tolist()}

@app.post("/reload_model")
def reload_model():
    load_model()
    return {"status": "reloaded"}
