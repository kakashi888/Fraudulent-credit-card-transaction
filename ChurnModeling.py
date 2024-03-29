# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:31:17 2019

@author: Rohit Soni
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:01:51 2019

@author: Rohit Soni
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
y=y.reshape(-1,1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])


labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train=y_train.reshape(-1,1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(output_dim=20,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=10,init='uniform',activation='relu',input_dim=20))

classifier.add(Dense(output_dim=5,init='uniform',activation='relu',input_dim=10))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid',input_dim=5))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)


y_pred = classifier.predict(X_test)

y_pred= (y_pred>0.4)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
