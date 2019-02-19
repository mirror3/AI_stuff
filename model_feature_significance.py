# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:45:16 2018

@author: haldes2
"""

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper
# Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split


def get_feature_significance(x_train,y_train,x_test,y_test,columns):
   
    
    # Some useful parameters which will come in handy later on
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    SEED = 0 # for reproducibility
    NFOLDS = 5 # set folds for out-of-fold prediction
    kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
    
    from helper import ModelHelper    
    
   
    
    # Parameters
        
    # Put in our parameters for said classifiers
    # Random Forest parameters
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 500,
         'warm_start': True, 
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features' : 'sqrt',
        'verbose': 0
    }
    
    # Extra Trees Parameters
    et_params = {
        'n_jobs': -1,
        'n_estimators':500,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }
    
    # AdaBoost parameters
    ada_params = {
        'n_estimators': 500,
        'learning_rate' : 0.75
    }
    
    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 500,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    }
    
    # Support Vector Classifier parameters 
    svc_params = {
        'kernel' : 'linear',
        'C' : 0.025
        }
    
    # Create 5 objects that represent our 4 models
    rf = ModelHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = ModelHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = ModelHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = ModelHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    svc = ModelHelper(clf=SVC, seed=SEED, params=svc_params)
    
 
    # Create our OOF train and test predictions. These base results will be used as new features
    et_oof_train, et_oof_test = helper.get_labels(et, x_train, y_train, x_test) # Extra Trees
    rf_oof_train, rf_oof_test = helper.get_labels(rf,x_train, y_train, x_test) # Random Forest
    ada_oof_train, ada_oof_test = helper.get_labels(ada, x_train, y_train, x_test) # AdaBoost 
    gb_oof_train, gb_oof_test = helper.get_labels(gb,x_train, y_train, x_test) # Gradient Boost
    svc_oof_train, svc_oof_test = helper.get_labels(svc,x_train, y_train, x_test) # Support Vector Classifier
    
    
    
    rf_feature = rf.feature_importances(x_train,y_train)
    et_feature = et.feature_importances(x_train, y_train)
    ada_feature = ada.feature_importances(x_train, y_train)
    gb_feature = gb.feature_importances(x_train,y_train)
    
#    Handcoded from above list
    
    rf_feature =[0.01042659,0.00166431,0.0005475,0.00014318,0.00298337,0.00034688
    ,0.00222215,0.02051383,0.00474225,0.02073242,0.00116082,0.00756095
    ,0.00323612,0.0601939,0.00077005,0.05961842,0.00220309,0.00532122
    ,0.00267934,0.01276776,0.01463749,0.0145585,0.01868941,0.01511414
    ,0.03040864,0.03565238,0.08007477,0.06610797,0.04514352,0.0395932
    ,0.01176934,0.01177531,0.00842211,0.0128518,0.01334165,0.01055991
    ,0.01047328,0.0047275,0.0128702,0.00729386,0.05595461,0.06999914
    ,0.00200907,0.02005596,0.01713513,0.00751636,0.02868192,0.03797604
    ,0.02298871,0.01662392,0.0131399,0.01460146,0.00941865]
    
    et_feature =[0.01328414,0.00352633,0.0015442,0.00030004,0.0074371,0.0058684
    ,0.0251283,0.04343197,0.00534433,0.02607072,0.00402104,0.02141162
    ,0.00494316,0.06430226,0.00386242,0.06055173,0.00699049,0.01104381
    ,0.00611286,0.01938496,0.00986196,0.00936905,0.02589242,0.00905692
    ,0.01391006,0.01344668,0.11123226,0.02169754,0.06984764,0.0227811
    ,0.01727597,0.01367694,0.0059404,0.00785247,0.01009598,0.00614055
    ,0.00463503,0.01023641,0.00853352,0.01828404,0.02138982,0.01695063
    ,0.00975726,0.03288239,0.04298873,0.01445468,0.03141432,0.03900584
    ,0.01783509,0.00787762,0.00678685,0.00656709,0.00776286]
    
    ada_feature =[0.022,0.002,0.002,0.002,0.006,0.,0.,0.002,0.008,0.026,0.002,0.01
    ,0.004,0.018,0.,0.022,0.002,0.022,0.004,0.026,0.022,0.05,0.004,0.03
    ,0.01,0.032,0.006,0.01,0.002,0.012,0.04,0.032,0.034,0.038,0.044,0.032
    ,0.052,0.024,0.032,0.012,0.016,0.04,0.006,0.008,0.022,0.028,0.03,0.022
    ,0.018,0.022,0.026,0.042,0.022]
    
    gb_feature = [0.00964416,0.00703517,0.,0.00114656,0.00758631,0.00257638
    ,0.00281657,0.0035492,0.00722141,0.00702195,0.00303237,0.00608315
    ,0.00498799,0.0071329,0.00144849,0.0107549,0.0031814,0.0049315
    ,0.00667917,0.00948474,0.04767588,0.04258683,0.00552803,0.04112653
    ,0.01448118,0.01578884,0.01231341,0.01403348,0.00501861,0.01084269
    ,0.01120256,0.03955186,0.03481087,0.0522866,0.05292039,0.05192869
    ,0.05570437,0.00729184,0.0210541,0.00881631,0.04172091,0.03033565
    ,0.00732274,0.00802642,0.00760243,0.00758147,0.04764953,0.05595386
    ,0.02431454,0.02142858,0.0346161,0.03655468,0.0256157]
    
    
    
    # Create a dataframe with features
    feature_dataframe = pd.DataFrame( {'features': columns,
         'Random Forest feature importances': rf_feature,
         'Extra Trees  feature importances': et_feature,
          'AdaBoost feature importances': ada_feature,
        'Gradient Boost feature importances': gb_feature
        })
    
    x = feature_dataframe['features'].values
    plt.title('Relative Feature Significance Benchmarking')
    plt.rcParams["figure.figsize"] = (20,10)
    plt.xticks(rotation='vertical')
    y = feature_dataframe['Random Forest feature importances'].values
    plt.bar(x,y,label='Random Forest feature importances')
    
    y = feature_dataframe['Extra Trees  feature importances'].values
    plt.bar(x,y,label='Extra Trees  feature importances')
    y = feature_dataframe['AdaBoost feature importances'].values
    plt.bar(x,y,label='AdaBoost feature importances')
    y = feature_dataframe['Gradient Boost feature importances'].values
    plt.bar(x,y,label='Gradient Boost feature importances')
    plt.legend(loc='upper left')
    plt.savefig('bencgmark_features.jpg')
    