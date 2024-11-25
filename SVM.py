import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
df= pd.read_csv("Social_Network_Ads.csv")
print(df.head())
print(df)
from sklearn.model_selection import train_test_split

train_set,test_set=train_test_split(df,test_size=0.30)
x_train=train_set.iloc[:,0:2].values
y_train=train_set.iloc[:,2].values
x_test=test_set.iloc[:,0:2].values
y_test=test_set.iloc[:,2].values
from sklearn.svm import SVC

model= SVC(kernel='rbf',random_state=30)
model.fit(x_train,y_train)
model.predict([[55,4]])
from sklearn import svm

clf=svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
from sklearn import metrics

print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("Recall ",metrics.recall_score(y_test,y_pred))
