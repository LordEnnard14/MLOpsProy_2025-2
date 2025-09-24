# src/service/serve.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd


#Conectar la base de datos local de MLFlow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Cargar el modelo desde el Model Registry de MLflow
MODEL_URI = "models:/telco-churn/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)

# FastAPI app
app = FastAPI(title="Telco Churn API", description="Servicio de predicción de cancelación de clientes", version="1.0")

# Clase que define el formato del input JSON
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: CustomerData):
    # Convertir el input a DataFrame
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])  # DataFrame 
    prediction = model.predict(df)
    return {
        "prediction": int(prediction[0])  # 0 = no churn, 1 = churn
    }
