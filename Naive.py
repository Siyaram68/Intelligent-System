import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
df= pd.read_csv('Social_Network_Ads.csv')
df.head()
x=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
train_set, test_set= train_test_split(df, test_size=0.25, random_state=30)
x_train=train_set.iloc[:,0:2].values
y_train=train_set.iloc[:,2].values
x_test=test_set.iloc[:,0:2].values
y_test=test_set.iloc[:,2].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(x_train, y_train)
from sklearn import metrics

y_pred=classifier.predict(x_test)
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("Precision",metrics.precision_score(y_test,y_pred))
print("RMSE:",metrics.mean_absolute_error(y_test,y_pred))
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
