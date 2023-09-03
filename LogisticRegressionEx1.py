# Logistic Regression

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\23rd,24th,25th - Classification\2.LOGISTIC REGRESSION CODE\logit classification.csv')

# Dividing the dataset into dependent variable (y) and independent variable (x)
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

# Splitting the Dataset into traning and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.25,random_state = 0)

# Feature Scaling 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Preprocessing is done upto this part

# Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# Predicting the test set Results

y_pred = classifier.predict(X_test)
y_pred

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Finding the model Accuracy

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac

# Getting the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
cr

# Find this a best model or not we check the bias and variance 

bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test,y_test)
variance
