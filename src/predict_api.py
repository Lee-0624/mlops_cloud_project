from fastapi import FastAPI
import mlflow, pandas as pd, os
from pydantic import BaseModel
from typing import List

app = FastAPI()
_ml_model = None

def load_model():
    global _ml_model
    try:
        _ml_model = mlflow.pyfunc.load_model("models:/seoul_temp/Production")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model not found or error loading model: {e}")
        _ml_model = None

class WeatherRow(BaseModel):
    nx:int; ny:int; hour:int; temp:float; rain:float

@app.on_event("startup")
def startup():
    load_model()

@app.post("/predict")
def predict(rows: List[WeatherRow]):
    if _ml_model is None:
        return {"error": "Model not loaded. Please train and register a model first."}
    df = pd.DataFrame([r.dict() for r in rows])
    preds = _ml_model.predict(df)
    return {"prediction": preds.tolist()}

@app.post("/reload_model")
def reload_model():
    load_model()
    return {"status": "reloaded"}
