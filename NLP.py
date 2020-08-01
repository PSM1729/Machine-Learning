#importing the dataset
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset= pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting= 3)

#Clean the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
word= []
for i in range(0, 1000):
    review= re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review= review.lower()
    review= review.split()
    ps= PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    word.append(review)
    
#Creating the bag Model
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
X= cv.fit_transform(word).toarray()
y= dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)