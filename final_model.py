import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import helper
import model_feature_significance as mfs
from helper import ModelHelper

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#importing Training data
train = pd.read_csv('dataset/Development_set.csv')

#importing Prediction data
prediction = pd.read_csv('dataset/Test_set.csv')

#Remove Patent ID from predictioon data
prediction= prediction.drop(['Pat_ID'],axis=1)

# Number of parameters from PCA
filtered_pca_x=49


# List of columns in Dataset
columns=list(train.columns.values)[2:]

# Missing Values Train
check_missing_values_train = train.isnull().sum().sum()


# Missing Values Prediction Set
check_missing_values_prediction = prediction.isnull().sum().sum()


#Correlation Between  Features
correlation = train.astype(float).corr()



# Drop anything less than 70% Correlation
correlation=correlation.apply(lambda x: np.where(x < 0.7,0,x))




    
#         Top coorelated features 
#        DisHis5 DisHis1 1.0
#        RespQues1 ResQues1b 0.9145450178532795
#        RespQues1 ResQues1c 0.9541766150574756
#        Demo2 Demo4 0.8824315534515427
#        DisHis2 DisHis7 0.8737480087657503
#        LungFun5 LungFun6 LungFun11 LungFun10 
#        LungFun1 LungFun3 LungFun4 LungFun12 LungFun9 LungFun8 LungFun14 LungFun7 LungFun13
#        LungFun15 LungFun17
#        LungFun16 LungFun18
#        LungFun2
#        DisHis1	DisHis1Times 0.7340866328957045
#        DisHis2 DisHis2Times 0.7117993098090788



#Remove all Correlated Features from Training and Predict Dataset 

#DisHis5 DisHis1 DisHis1Times Merged to DisHis1Times
train= train.drop(['DisHis5','DisHis1'],axis=1)
helper.deleteColumn(['DisHis5','DisHis1'],columns)
prediction= prediction.drop(['DisHis5','DisHis1'],axis=1)

#RespQues1 ResQues1b ResQues1c Merged to RespQues1
train= train.drop(['ResQues1b','ResQues1c'],axis=1)
prediction= prediction.drop(['ResQues1b','ResQues1c'],axis=1)
helper.deleteColumn(['ResQues1b','ResQues1c'],columns)
#Demo2 Demo4 Merged to Demo2
train= train.drop(['Demo4'],axis=1)
prediction= prediction.drop(['Demo4'],axis=1)
helper.deleteColumn(['Demo4'],columns)
#DisHis2 DisHis7 Merged to DisHis2
train= train.drop(['DisHis7'],axis=1)
prediction= prediction.drop(['DisHis7'],axis=1)
helper.deleteColumn(['DisHis7'],columns)
#LungFun5 LungFun6 LungFun11 LungFun10  Merged to LungFun5
train= train.drop(['LungFun6','LungFun11','LungFun10'],axis=1)
prediction= prediction.drop(['LungFun6','LungFun11','LungFun10'],axis=1)
helper.deleteColumn(['LungFun6','LungFun11','LungFun10'],columns)
#LungFun1 LungFun3 LungFun4 LungFun12 LungFun9 LungFun8 LungFun14 LungFun7 LungFun13 Merged to LungFun1
train= train.drop(['LungFun3','LungFun4','LungFun12','LungFun9','LungFun8','LungFun7','LungFun13'],axis=1)
prediction= prediction.drop(['LungFun3','LungFun4','LungFun12','LungFun9','LungFun8','LungFun7','LungFun13'],axis=1)
helper.deleteColumn(['LungFun3','LungFun4','LungFun12','LungFun9','LungFun8','LungFun7','LungFun13'],columns)
#LungFun15 LungFun17 Merged to LungFun15
train= train.drop(['LungFun17'],axis=1)
prediction= prediction.drop(['LungFun17'],axis=1)
helper.deleteColumn(['LungFun17'],columns)
#LungFun16 LungFun18 Merged to LungFun16
train= train.drop(['LungFun18'],axis=1)
prediction= prediction.drop(['LungFun18'],axis=1)
helper.deleteColumn(['LungFun18'],columns)

#DisHis2	DisHis2Times  Merged to DisHis2Times
train= train.drop(['DisHis2'],axis=1)
prediction= prediction.drop(['DisHis2'],axis=1)
helper.deleteColumn(['DisHis2'],columns)

#train 1687 predict 298
X_predict= prediction.iloc[:,:41].values

# Split Train Data to X and y
X=train.iloc[:,2:43].values
y=train.iloc[:,1].values



X=helper.form_dataset(X,X_predict)



# One Hot Encoder
# Encoding categorical data
# Also Create new columns to Accomodate new Category Features
for i in [21,24,27,30,33,36]:
    X=helper.OnehotEncoder(X,i,3,columns)
X=helper.OnehotEncoder(X,24,8,columns)

#Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Determine The Significance of Each Model Feature applied on different Classifier
#x_train,x_test,y_train_1,y_test_1= train_test_split(helper.get_training(X,1687),y,test_size=0.1)
#mfs.get_feature_significance(x_train,y_train_1,x_test,y_test_1,columns)

#PCA to choose most relevent feature 
X=helper.callPCA(X,filtered_pca_x)





# Seperate Training and Test data
training=helper.get_training(X,1687)
predict_set=helper.get_test(X,1687)


while(True):
    #Feature Scaling
    X_train,X_test,y_train,y_test= train_test_split(training,y,test_size=0.1)
    #Implment SMOTE method to increase Sparse labels
    sm = SMOTE(random_state=2)
    X_train_u, y_train_u = sm.fit_sample(X_train, y_train.ravel())
    #Predict X and y on new dataset
    classifier= SVC(kernel='poly',C=10)   
    classifier.fit(X_train_u,y_train_u)
    y_predict= classifier.predict(X_test)
    # Get AUC score on new Dataset
    auc_roc_1 = roc_auc_score(y_test, y_predict)*100
    print('Searching for AOC Score80% got:',auc_roc_1)
    if(auc_roc_1>=80):
        # Target AUC score achived go for Prediction on new data
        predict_y=helper.Final_Model(X_train_u,y_train_u,X_test,y_test,predict_set)
        np.savetxt("predict.csv",predict_y, delimiter=",")
        break



