from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="Telco Churn Predictor")

# Modelo se carga desde el Model Registry
MODEL_URI = "models:/telco-churn/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Esquema para recibir datos (lista de listas de features)
class PredictionRequest(BaseModel):
    features: list[list[float]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    X = pd.DataFrame(request.features)
    y_pred = model.predict(X)
    return {"predictions": y_pred.tolist()}
