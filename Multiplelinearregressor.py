#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
dataset= pd.read_csv('50_Startups.csv')
x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, 4].values

#Categoring the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder= LabelEncoder()
x[:, 3]= labelencoder.fit_transform(x[:, 3])
onehotencoder= OneHotEncoder(categorical_features = [3])
x= onehotencoder.fit_transform(x).toarray()

#Avoid Dummy Variable trap
x= x[:, 1:]

#Training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split( x, y, test_size= 0.2, random_state= 0)

#building the model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the value
y_pred= regressor.predict(x_test)

#Backward Elimination with p-values
import statsmodels.api as sm
def BackwardElimination(x, sl):
    numlen= len(x[0])
    for __ in range(0 , numlen):
        regressor_ols= sm.OLS(y,x).fit()
        maxval= max(regressor_ols.pvalues).astype(float)
        if maxval > sl :
            for j in range(0, numlen-1):
                if(regressor_ols.pvalues[j].astype(float) == maxval):
                    np.delete(x,j,1)
    regressor_ols.summary()
    return x
            
    
 
SL= 0.05
x_opt= x[:, [0,1,2,3,4]]
x_modelled= BackwardElimination(x_opt, SL)