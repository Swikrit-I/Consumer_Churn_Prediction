import numpy as np
import pandas as pd



#Function that replaces string to to numeric value and nan if necessary and perforom mean imputation on it
def cleaning(data):
    #to_numeric only receives 1d array but later we will pass 2d array through column transformer in preprocessing pipeline
    data=np.array(data).flatten()
    
    #changes string data into float and replaces some unchangable string into nan value
    data=pd.to_numeric(data,errors="coerce")

    from sklearn.impute import SimpleImputer
    imputer=SimpleImputer(strategy="mean")

    #np.array(data).reshape(-1,1), it changes the shape of 1d panda into 2d numpy array
    data=imputer.fit_transform(np.array(data).reshape(-1,1))

    returning_data=(pd.DataFrame(data))
    return returning_data


def binary(data):
    maping={"Yes":1,"No":0,"Male":1,"Female":0}
    data=data.replace(maping)

    return data
