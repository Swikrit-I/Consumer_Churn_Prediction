import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score,accuracy_score
from packages.utilities import binary


pipeline=joblib.load("/Users/swikritsubedi/Desktop/Python/Consumer Churn Prediction/models/pipeline.joblib")
df=pd.read_csv("/Users/swikritsubedi/Desktop/Python/Consumer Churn Prediction/data/Raw/raw.csv")

#loading data for testing
y_test=df.loc[3000:,"Churn"]
x_drop=df.drop(["Churn","customerID"],axis=1)
x_test=x_drop.iloc[3000:,:]

#setting standards for pipeline

#making prediction
y_predict=pipeline.predict(x_test)
y_predict=pd.DataFrame(y_predict)
#calculating accuracy and f1 score
print(accuracy_score(y_predict,y_test))
