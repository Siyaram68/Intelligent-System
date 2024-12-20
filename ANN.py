import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
df=pd.read_csv("diabetes.csv")
print(df.shape)
df.describe().transpose()
target_column=['Outcome']
predictors=list(set(list(df.columns))-set(target_column))
df[predictors]=df[predictors]/df[predictors].max()
df.describe().transpose()
X=df[predictors].values
y=df[target_column].values
X_train,X_test,y_train,y_test=train_test_split(X,y.ravel(),test_size=0.3,random_state=1)
print(X_train.shape)
print(X_test.shape)
mlp=MLPClassifier(hidden_layer_sizes=(8,8,8),activation='relu',solver='adam',max_iter=500)
mlp.fit(X_train,y_train)
predict_train=mlp.predict(X_train)
predict_test=mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predict_test)
print(accuracy)
