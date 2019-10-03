# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:08:31 2019

@author: Rohit Soni
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import imblearn

dataset = pd.read_csv('creditcard.csv')
X = dataset.drop("Class",axis=1).values
y = dataset["Class"].values

from imblearn.under_sampling import NearMiss
rus=NearMiss(random_state=0)
X,y=rus.fit_sample(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import r2_score
acc=sklearn.metrics.accuracy_score(y_test, y_pred) 
score=r2_score(y_test,y_pred)

from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
