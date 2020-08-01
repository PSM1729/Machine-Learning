#Importing the dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
dataset= pd.read_csv('Social_Network_Ads.csv')
x= dataset.iloc[:, [2,3]].values
y= dataset.iloc[:, 4].values

#Creating Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)

#Fitting the training set to the Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#Predicting The Value
y_pred= classifier.predict(x_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#Visualing The Training set
from matplotlib.colors import ListedColormap
x_set, y_set= x_train, y_train
x1,x2= np.meshgrid(np.arange(x_set[:, 0].min()-1, x_set[:, 0].max()+1, step= 0.5),
                   np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), 
             alpha= 0.75, cmap= ListedColormap(('blue', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j, 0], x_set[y_set ==j,1],
                c= ListedColormap(('blue', 'green'))(i), label = j)
plt.title('Logistic Regression(Train set)')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

#Visualing The Test Set
from matplotlib.colors import ListedColormap
x_set, y_set= x_test, y_test
x1,x2= np.meshgrid(np.arange(x_set[:, 0].min()-1, x_set[:, 0].max()+1, step= 0.01),
                   np.arange(x_set[:, 1].min()-1, x_set[:, 1].max()+1, step= 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha= 0.75, cmap= ListedColormap(('blue', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0], x_set[y_set== j,1],
                c= ListedColormap(('blue', 'green'))(i), label = j)
plt.title('Logistic Regression(Test set)')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()