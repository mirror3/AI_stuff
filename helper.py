#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:41:46 2018

@author: sandi
"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import statsmodels.formula.api as sm
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.decomposition import PCA

def OnehotEncoder(X_split,row,size,columns):
    labelencoder_X = LabelEncoder()
    X_split[:, row] = labelencoder_X.fit_transform(X_split[:, row])
    onehotencoder = OneHotEncoder(categorical_features = list([row]))
    X_split = onehotencoder.fit_transform(X_split).toarray()
    np.delete(X_split,row+size,1)
    column_name=columns[row]
    for i in range(1,size):
        columns.insert(0,column_name+'_'+str(i))
    del columns[row+size-1]
    return X_split[:,1:len(columns)+1]

def deleteColumn(columns_list,columns):
    for i in columns_list:
        if i in columns:
            columns.remove(i)
    



def backwardElimination(x,x_t, SL,filtered_pca_x,y_train):
    numVars = len(x[0])
    val = x.shape[0]
    temp = np.zeros((val,filtered_pca_x)).astype(int)
    temp_test = np.zeros((169,filtered_pca_x)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    temp_test[:,j] = x_t[:, j]
                    x = np.delete(x, j, 1)
                    x_t = np.delete(x_t, j, 1)
                    tmp_regressor = sm.OLS(y_train, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        x_rollback_test = np.hstack((x_t, temp_test[:,[0,j]]))
                        x_rollback_test = np.delete(x_rollback_test, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback,x_rollback_test
                    else:
                        continue
    regressor_OLS.summary()
    return x,x_t

def randomSearch(model,param_grid,X_train,y_train):
    
    grid_search =RandomizedSearchCV(model, param_grid, n_iter=100,
                            n_jobs=4, verbose=2, cv=2,
                            scoring='neg_log_loss', refit=False, random_state=42)
    grid_search.fit(X_train, y_train)
    print("Model Parameters: {}".format(grid_search.best_params_))
    print(grid_search.best_score_)
    
def gridSearch(model,param_grid,X_train,y_train,X_test,y_test):
    
    gridsearch = GridSearchCV(model, param_grid)

    gridsearch.fit(X_train, y_train,
        eval_set = [(X_test, y_test)],
        eval_metric = ['auc', 'binary_logloss'],
        early_stopping_rounds = 5)
    print("Model Parameters: {}".format(grid_search.best_params_))
    print(grid_search.best_score_)

 
    
def SVCRandomSearch(X_train,y_train):
    model =SVC(kernel='rbf',random_state=0,probability=True)
    param_grid = {'C':range(1,1000), 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
    randomSearch(model,param_grid,X_train,y_train)
    
#Test Accuracy: 0.8402366863905325
#Cross Validation Accuracy: 0.8735250833225123 Variance(%)): 0.2154070360518396
def KNNRandomSearch(X_train,y_train):
    param_grid = {'n_neighbors':range(2,100), 'p':[2,3], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
    model =KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    randomSearch(model,param_grid,X_train,y_train)
    
#Test Accuracy: 0.9112426035502958
#Cross Validation Accuracy: 0.8656679640602609 Variance(%)): 1.3604045136127765
def LRRandomSearch(X_train,y_train):
    param_grid = {'fit_intercept':[True,False], 'max_iter':range(1,100), 'random_state':[True, False]}
    model =LogisticRegression(fit_intercept=True,max_iter=10,random_state=0)
    randomSearch(model,param_grid,X_train,y_train)
    
    
###Test Accuracy:  0.8402366863905325
###Cross Validation Accuracy: 0.8735175438596491 Variance(%)): 1.5225172029982694
def XGBBoostRandomSearch(X_train,y_train):
    model = XGBClassifier(learning_rate =0.1, n_estimators=1000,max_depth=5,min_child_weight=1,gamma=1,reg_lambda= 0.55,reg_alpha=0.48,subsample=1,colsample_bytree=0.6,objective= 'binary:logistic', nthread=8,scale_pos_weight=1,seed=27)
    param_grid = {
        'learning_rate':[0.10, 0.125, 0.15, 0.175, 0.2],
        'gamma': [.05, .01, .03,.05,.07,.09,1],          # Depth of the tree
        'max_depth': [3,5,7,9,12,15,17,25],  # Minimum number of samples required to split an internal node
        'min_child_weight': [1,3,5,7],    # Minimum number of samples in a leaf,
        'subsample': [.6,.7,.8,.9,1],  # Number of features for each tree
        'colsample_bytree': [.6,.7,.8,.9,1] ,        # Depth of the tree
        'reg_alpha':[0.01,0.02,0.03,0.04,0.05,0.060000000000000005,0.06999999999999999,0.08,0.09,0.09999999999999999,0.11,0.12,0.13,0.14,0.15000000000000002,0.16,0.17,0.18000000000000002,0.19,0.2,0.21000000000000002,0.22,0.23,0.24000000000000002,0.25,0.26,0.27,0.28,0.29000000000000004,0.3,0.31,0.32,0.33,0.34,0.35000000000000003,0.36000000000000004,0.37,0.38,0.39,0.4,0.41000000000000003,0.42000000000000004,0.43,0.44,0.45,0.46,0.47000000000000003,0.48000000000000004,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.5700000000000001,0.5800000000000001,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.6900000000000001,0.7000000000000001,0.7100000000000001,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.8200000000000001,0.8300000000000001,0.8400000000000001,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.9400000000000001,0.9500000000000001,0.9600000000000001,0.97,0.98,0.99],
        'reg_lambda':[0.01,0.02,0.03,0.04,0.05,0.060000000000000005,0.06999999999999999,0.08,0.09,0.09999999999999999,0.11,0.12,0.13,0.14,0.15000000000000002,0.16,0.17,0.18000000000000002,0.19,0.2,0.21000000000000002,0.22,0.23,0.24000000000000002,0.25,0.26,0.27,0.28,0.29000000000000004,0.3,0.31,0.32,0.33,0.34,0.35000000000000003,0.36000000000000004,0.37,0.38,0.39,0.4,0.41000000000000003,0.42000000000000004,0.43,0.44,0.45,0.46,0.47000000000000003,0.48000000000000004,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.5700000000000001,0.5800000000000001,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.6900000000000001,0.7000000000000001,0.7100000000000001,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.8200000000000001,0.8300000000000001,0.8400000000000001,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.9400000000000001,0.9500000000000001,0.9600000000000001,0.97,0.98,0.99]
        }        
    randomSearch(model,param_grid,X_train,y_train)    
#    gridSearch(model,param_grid,X_train,y_train,X_test,y_test) 
    
    
    
##Test Accuracy: 0.8520710059171598
##Cross Validation Accuracy: 0.8761632688395447 Variance(%)): 0.545088202983163
def RFRandomSearch(X_train,y_train):
    param_grid = {'fit_intercept':[True,False], 'max_iter':range(1,100), 'random_state':[True, False]}
    model = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
    param_grid = {    
        'n_estimators':[120,300,500,800,1200],          # Depth of the tree
        'max_depth':[5,8,15,30,None],  # Minimum number of samples required to split an internal node
        'min_samples_split':[2,5,10,15,100],    # Minimum number of samples in a leaf,
        'min_samples_leaf':[1,2,5,10],  # Number of features for each tree
        'max_features':['log2','sqrt','auto']        # Depth of the tree
       } 
    randomSearch(model,param_grid,X_train,y_train)    
    
#Test Accuracy: 0.8816568047337278
#Cross Validation Accuracy: 0.8695712791913559 Variance(%)): 0.5601486325388777
def ExtraTreeRandomSearch(X_train,y_train):
    
    param_grid = {  
        'n_estimators':[120,300,500,800,1200],
        'max_features': [0.6, 0.75, 0.9],  # Number of features for each tree
        'max_depth': [5, 15, 25],          # Depth of the tree
        'min_samples_split': [5, 10, 50],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [5, 10, 50]    # Minimum number of samples in a leaf
        }
    
        ############ Create a ExtraTreesClassifier with initial parameters ########
    model = ExtraTreesClassifier(    
        n_estimators = 100,             # Number of trees
        max_features = 0.8,             # Number of features for each tree
        max_depth = 10,                 # Depth of the tree
        min_samples_split = 2,          # Minimum number of samples required to split
        min_samples_leaf = 1,           # Minimum number of samples in a leaf
        min_weight_fraction_leaf = 0,   # Minimum weighted fraction of the input samples required to be at a leaf node. 
        criterion = 'gini',             # Use gini, not going to tune it
        random_state = 27,
        n_jobs = -1)

    randomSearch(model,param_grid,X_train,y_train)    
    
    
def model_performance(classifier,X_trained,y_train,X_tested,y_test):
    classifier.fit(X_trained,y_train)
    y_predict= classifier.predict(X_tested)
    accuracies= cross_val_score(estimator= classifier,X= X_trained,y= y_train, cv=10)
    c_matrix = confusion_matrix(y_test,y_predict)    
    auc_roc_1 = roc_auc_score(y_test, y_predict)
    print('AOC',auc_roc_1)
#    print('Test Accuracy:',classifier.score(X_tested,y_test)*100)
    print('Cross Validation Accuracy:',accuracies.mean(),'Variance(%)):',accuracies.std()*100)
    print('c_matrix',c_matrix)
    P=c_matrix[0,0]/(c_matrix[0,0]+c_matrix[0,1])*100
    R=c_matrix[0,0]/(c_matrix[0,0]+c_matrix[1,0])*100
    S=c_matrix[1,1]/(c_matrix[1,0]+c_matrix[1,1])*100
    F1=2*((P*R)/(P+R))
    print('Scores: P:{} R:{} S:{} F1:{}'.format(P,R,S,F1))

      
def callPCA(X,filtered_pca_x):
    pca = PCA(n_components=filtered_pca_x)
    X= pca.fit_transform(X)
    explained_variance_percent=pca.explained_variance_ratio_*100
    print('explained_variance_percent',explained_variance_percent)
    return X
    
    

def get_labels(clf, x_train, y_train, x_test):
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    NFOLDS = 5 
    SEED = 0 # for reproducibility
    kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    
 # Class to extend the Sklearn classifier
class ModelHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        
   
def unpack_models(classifier,x_train,y_train,x_test):
    train_size=x_train.shape[0]
    test_size=x_test.shape[0]
    classifier.fit(x_train,y_train)
    train_y=[None]*train_size
    test_y=[None]*test_size
    print(np.reshape(x_train[1],(x_train.shape[1],1)))
    for i in range(train_size):
        train_y[i]=int(classifier.predict(np.reshape(x_train[i],(1,-1))))
    for i in range(test_size):
        test_y[i]=int(classifier.predict(np.reshape(x_test[i],(1,-1))))
    return np.reshape(train_y,(train_size,1)),np.reshape(test_y,(test_size,1))

def form_dataset(x,y):
    return np.concatenate((x,y),axis=0)

def get_training(x,index):
    return x[:index,:]

def get_test(x,index):
    return x[index:,:]
       
        
     
def Final_Model(X_train,y_train,X_test,y_test,predict_set):
    svc= SVC(kernel='poly',C=10)
    model_performance(svc,X_train,y_train,X_test,y_test)
    predict_y=svc.predict(predict_set)
    return predict_y

  
#def build_seconday_classifier(X_train,y_train,X_test,y_test):
#    
#    xg = XGBClassifier(learning_rate =0.01, n_estimators=500,max_depth=15,min_child_weight=1,gamma=.05,reg_lambda= 0.29,reg_alpha=0.34,subsample=1,colsample_bytree=1,objective= 'binary:logistic', nthread=8,scale_pos_weight=1,seed=27)
#    knn = KNeighborsClassifier(n_neighbors=15,metric='minkowski',p=2,algorithm='auto')
#    
#    et = ExtraTreesClassifier(    
#            n_estimators = 100,             # Number of trees
#            max_features = 0.9,             # Number of features for each tree
#            max_depth = 25,                 # Depth of the tree
#            min_samples_split = 10,          # Minimum number of samples required to split
#            min_samples_leaf = 5,           # Minimum number of samples in a leaf
#            min_weight_fraction_leaf = 0,   # Minimum weighted fraction of the input samples required to be at a leaf node. 
#            criterion = 'gini',             # Use gini, not going to tune it
#            random_state = 27,
#            n_jobs = -1)
#    
#    svc= SVC(kernel='poly',C=10)
#    
#    
#    xg_train,xg_test=helper.unpack_models(xg,X_train,y_train,X_test)
#    knn_train,knn_test=helper.unpack_models(knn,X_train,y_train,X_test)
#    et_train,et_test=helper.unpack_models(et,X_train,y_train,X_test)
#    svc_train,svc_test=helper.unpack_models(svc,X_train,y_train,X_test)
#    
#    x_train_secondary = np.concatenate(( xg_train, knn_train, et_train, svc_train), axis=1)
#    x_test_secondary = np.concatenate(( xg_test, knn_test, et_test, svc_test), axis=1)
#    
#    #helper.XGBBoostRandomSearch(X_train,y_train)
#    
#    seconary_classifier= XGBClassifier(learning_rate =0.01, 
#                              n_estimators=500,
#                              max_depth=15,
#                              min_child_weight=1,
#                              gamma=.05,
#                              reg_lambda= 0.29,
#                              reg_alpha=0.34,
#                              subsample=.6,
#                              colsample_bytree=1,
#                              objective= 'binary:logistic',
#                              nthread=8,
#                              scale_pos_weight=1,
#                              seed=27)
#    
#    helper.model_performance(seconary_classifier,x_train_secondary,y_train,x_test_secondary,y_test)
