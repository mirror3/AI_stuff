# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



#importing Training data
train_dataset = pd.read_csv('dataset\Development_set.csv')

#importing Test data
#validation_dataset = pd.read_csv('Test_set.csv')

# Split Train Data to X and y
X=train_dataset.iloc[:,2:64].values
y=train_dataset.iloc[:,1].values

def scale_column(i):
    scaler_train=StandardScaler()
    X[:,i]=np.reshape(scaler_train.fit_transform(np.reshape(X[:,i],(len(X),1))),(len(X)))

# Split Train Test to X and y
#X_test=validation_dataset.iloc[:,1:63]

# Missing Values Train
check_missing_values_train = train_dataset.isnull().sum().sum()

# Missing Values Test
#check_missing_values_test = validation_dataset.isnull().sum().sum()
array_scaling_index=[1,2,3,5,7,9,11,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,10,31,32,33,34,35,37,38,40,41,42,43,44,46,48,51,52,53,54,55,56,57,58,59]
for i in array_scaling_index:
    scale_column(i)




#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=15)
X= pca.fit_transform(X)
explained_variance=pca.explained_variance_ratio_

# LDA    
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda = LDA(n_components=10)
#X=lda.fit_transform(X,y)
    
##Kernal PCA
#from sklearn.decomposition import KernelPCA
#kpca = KernelPCA(n_components=10,kernel='rbf')
#X= kpca.fit_transform(X)

# Train Test Split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 15))
classifier.add(Dropout(rate=0.2))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.2))
#
## Adding the second hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform',classifier.add(Dropout(rate=0.2))



# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 150)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('Accuracy%:',(cm[0,0]+cm[1,1])/cm.sum()*100)


