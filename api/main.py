import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Literal
from packages import preprocessing
from pathlib import Path


#creating instances for data validation
class Consumer_churn(BaseModel):
    customerID:str    
    gender: Literal["Male","Female"]
    SeniorCitizen: Literal[0,1]
    Partner: Literal["Yes","No"]
    Dependents: Literal["Yes","No"]
    tenure: int =Field(...,gt=0)
    PhoneService: Literal["Yes","No"]
    MultipleLines: Literal["Yes","No","No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes","No"]
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)']
    MonthlyCharges:float
    TotalCharges:str|float|int

app=FastAPI()

#setting path for model
path=Path(__file__).resolve().parent.parent/"models"/"pipeline.joblib"
pipeline=joblib.load(path)

@app.get("/")
def home():
    return{
        "status":"Activate",
        "message":"Welcome to churn prediction api"
    }
@app.post("/predict")
def predict(data:Consumer_churn):
    data_dict=data.model_dump()
    df=(pd.DataFrame([data_dict])).drop("customerID",axis=1)
    prediction=pipeline.predict(df)[0]

    return{
        "Coustomer Id":data.customerID,
        "Churn Condition":prediction
    }
