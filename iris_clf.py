from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=300,random_state=0).fit(X, y)
print(clf.predict(X[:2, :]))

print(clf.predict_proba(X[:2, :]))


print(clf.score(X, y))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris=sns.load_dataset('iris')
sp_list=[0]*150
for i in range(0,150):
    if i>=50:
        sp_list[i]=1
iris['label']=sp_list
iris.drop('species',axis=1,inplace=True)
print(iris)
print(iris.info())
print(iris.describe())

X=iris.drop('label',axis=1)
y=iris['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3333, random_state=99)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

