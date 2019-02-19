# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:14:53 2018

@author: sriniv11
"""

import pandas 

names = ['Sample code number',
         'Clump Thickness',
         'Uniformity of Cell Size',
         'Uniformity of Cell Shape',
         'Marginal Adhesion',
         'Single Epithelial Cell Size',
         'Bare Nuclei',
         'Bland Chromatin',
         'Normal Nucleoli',
         'Mitoses',
         'Class']

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

dataset =  pandas.read_csv(url,names=names)

print(dataset.shape)
print(dataset.head(10))
print(dataset['Class'])
print(dataset.groupby('Class').size())


from sklearn import model_selection
array = dataset.values


print("this is it ",array[0][5]+1)


for i in range(len(array)):
    for j in range(len(array[0])):
        if(array[i][j]=='?'):
            array[i][j]=0
        array[i][j] = int(array[i][j])
'''           
x=0
for i in array:
    for j in i:
        print("j = ",j)
        print(int(j)+1)
        if(j!=int(j)):
            
            print("WTF")
            x=1
            break
    if(x==1):
        break
        
'''
X= array[:,0:10]
Y= array[:,10]

Y =Y.astype('int')

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size = .9)

'''

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

'''
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier()
classifier.fit(X_train,Y_train)
prediction = classifier.predict(X_test)
 
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction))
print(confusion_matrix(Y_test,prediction))
