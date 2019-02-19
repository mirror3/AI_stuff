# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:02:14 2018

@author: sriniv11
"""
import matplotlib.pyplot as plt

import csv
import numpy as np

import pandas as pd

x= np.arange(0,2 * np.pi,0.1)

file_con = pd.read_csv('train.csv')

from sklearn.preprocessing import PolynomialFeatures

#file_con = file_con.drop('Survived',1)

file_con = file_con[np.isfinite(file_con['Age'])]
print(file_con)

file_con = file_con.drop('Cabin',1)
file_con = file_con.drop('Embarked',1)
file_con = file_con.drop('Ticket',1)
file_con = file_con.drop('Name',1)
file_con = file_con.drop('Sex',1)
print(file_con.columns.tolist())

print(file_con)
p = PolynomialFeatures(degree=3).fit(file_con)
print (p.get_feature_names(file_con.columns), len(p.get_feature_names(file_con.columns)))
print('p =',p)

features = pd.DataFrame(p.transform(file_con), columns=p.get_feature_names(file_con.columns))
print(features)

file_con_array = file_con.values

file_con_array = np.delete(file_con_array,0,1)

file_con_array = np.delete(file_con_array,2,1)

file_con_array = np.delete(file_con_array,6,1)

file_con_array = np.delete(file_con_array,7,1)


for i in range(len(file_con_array)):
    if(file_con_array[i][2] == 'male'):
        file_con_array[i][2] = 1
    else:
        file_con_array[i][2] = 0
    if(file_con_array[i][7] == 'S'):
        file_con_array[i][7] = 0
    elif(file_con_array[i][7] == 'C'):
        file_con_array[i][7] = 1
    else:
        file_con_array[i][7] = 2
        
    file_con_array[i][6] = int(file_con_array[i][6])
        
    
print(file_con_array[:10,:])
print(file_con_array[0])
print(np.shape(file_con_array))


from sklearn import model_selection

X = file_con_array[:,2:]
Y = file_con_array[:,0]

Y =Y.astype('int')


X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size = .3)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


classifier =  RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train,Y_train)
prediction = classifier.predict(X_test)
 
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction))
print(confusion_matrix(Y_test,prediction))

#y= np.sin(x)
'''
y= x
plt.subplot(2,1,1)
plt.pie(x,y)

y = np.cos(x)
plt.subplot(2,1,2)
plt.plot(x,y,color = 'blue', linewidth = 2)
plt.xlabel('x')
plt.ylabel('func(x)')


plt.show()
'''