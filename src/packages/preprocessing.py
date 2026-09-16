import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from packages.utilities import cleaning,binary
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,FunctionTransformer
from sklearn.compose import ColumnTransformer

#Trasforming function
custom_cleaning=FunctionTransformer(cleaning,validate=False)
binary_Encoding=FunctionTransformer(binary,validate=False)

#Selecting Features for feature encoding
binary_encoding=["gender","Partner","Dependents","PhoneService","PaperlessBilling"]
one_hot_encoding=["MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaymentMethod"]
ordinal_encoding=["Contract"]
scalerization=["tenure","MonthlyCharges"]

#ranking order for onehot encoding
contract_order=["Month-to-month","One year","Two year","TotalCharges"]

#Building function that has preprocessing pipeline
def preprocessing_pipeline():
    imputation_pipeline=Pipeline(
        steps=[("Cleaning",custom_cleaning)]
    )

    binary_encoding_pipeline=Pipeline(
        steps=[
            ("Binary",binary_Encoding),
            ("passthrough","passthrough")
            ]
    )
    onehot_encoding_pipeline=Pipeline(
        steps=[
            ("OneHot",OneHotEncoder(drop="first",sparse_output=False))
            ]
    )
    ordinal_encoding_pipeline=Pipeline(
        steps=[
            ("Ordinal",OrdinalEncoder(categories=[contract_order]))
        ]
    )
    scaling_pipeline=Pipeline(
        steps=[
            ("Scalerization",StandardScaler())
        ]
    )

    processor=ColumnTransformer(
        transformers=[
            ("imputation",imputation_pipeline,["TotalCharges"]),
            ("binary",binary_encoding_pipeline,binary_encoding),
            ("onehot",onehot_encoding_pipeline,one_hot_encoding),
            ("ordinal",ordinal_encoding_pipeline,ordinal_encoding),
            ("scalerization",scaling_pipeline,scalerization)
        ],
        remainder="passthrough"
    )

    return processor

def full_pipeline(model):

    processor_a=preprocessing_pipeline()
    final_pipeline=Pipeline(
        steps=[
            ("Processor",processor_a),
            ("model",model)
        ]
    )
    return final_pipeline
