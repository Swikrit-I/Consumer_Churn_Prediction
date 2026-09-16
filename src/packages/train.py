import numpy as np
import pandas as pd
from packages.preprocessing import full_pipeline,preprocessing_pipeline
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

#Importing raw data
df=pd.read_csv("/Users/swikritsubedi/Desktop/Python/Consumer Churn Prediction/data/Raw/raw.csv")

#selecting features for training
churn_data=df["Churn"]
data=df.drop(["customerID","Churn"],axis=1)

#selecting training data
x_train=data.iloc[:3000,:]
y_train=churn_data.iloc[:3000]


model=LogisticRegression()

final_pipeline=full_pipeline(model)
final_pipeline.fit(x_train,y_train)

#setting relative path
path=Path(__file__).resolve().parent.parent.parent/ "models"/"pipeline.joblib"
#saving the model
joblib.dump(final_pipeline,path)