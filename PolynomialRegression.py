# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Linear Rgeression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
polyreg= PolynomialFeatures(degree= 4)
polyreg.fit(X)
x_poly= polyreg.fit_transform(X,y)
linreg2= LinearRegression()
linreg2.fit(x_poly,y)

#Visualing Linear Regression
plt.scatter(X,y, color='orange')
plt.plot(X,linreg.predict(X), color='green')
plt.title('Truth')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualing Polynomial Feature
X_grid= np.arange(min(X), max(X), 0.1)
X_grid= X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color='orange')
plt.plot(X_grid,linreg2.predict(polyreg.fit_transform(X_grid)), color='green')
plt.title('Truth')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
