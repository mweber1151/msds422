#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:48:40 2020

@author: msweber
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime       
from sklearn.decomposition import PCA


class MNIST:
    
    def __init__(self, data):
        self.X, self.y = data
        self.RFC = None
        self.valAcc = None

    def tts(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y,
                                                            test_size=0.2
                                                            )
        data =  (X_train, X_test, y_train, y_test)
        return data

        
    def randForClass(self, data, n_est):
        X_train, X_test, y_train, y_test = data
        start_time = datetime.now()
        print("Start Time: " + str(start_time))
        RFC = RandomForestClassifier(n_estimators=n_est, 
                                     n_jobs=-1
                                     )
        RFC.fit(X_train, y_train)
        y_pred = RFC.predict(X_test)
        end_time = datetime.now()
        print("End Time: " + str(end_time))
        elapsed_time = end_time - start_time
        print("Elapsed Time: " + str(elapsed_time))
        valAcc = accuracy_score(y_test, y_pred)
        print("Validation Accuracy: " + str(valAcc))
        return RFC, valAcc, elapsed_time
    
    def cv(self, mult, n_est):
        RF = None
        VA = 0
        ET = 0
        it = 0
        for n in range(0, mult):
            data = self.tts()
            RFC, valAcc, elapsed_time = self.randForClass(data, n_est)
            if valAcc > VA:
                RF = RFC
                VA = valAcc
                ET = elapsed_time
            it += 1
            print("Iteration #" + str(it) + " completed")   
        return RF, VA, ET
 
if __name__ == '__main__':
    train = pd.read_csv("/home/msweber/Documents/msds422/train.csv")
    #Extract target variable
    y = train['label'].values
    train1 = train.drop(['label'], axis = 1)
    X = train1.values
    test = pd.read_csv("/home/msweber/Documents/msds422/test.csv")
    X_val = test.values
    start_pca = datetime.now()
    print("Start PCA Time: " + str(start_pca))
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    X_val_reduced = pca.transform(X_val)
    end_pca = datetime.now()
    print("End PCA Time: " + str(end_pca))
    elapsed_pca = end_pca - start_pca
    print("Elapsed PCA Time: " + str(elapsed_pca))
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    print(pca.n_components_)
    mn = MNIST((X_reduced, y))
    RF, VA, ET = mn.cv(10, 500)
    y_val = RF.predict(X_val_reduced)  
    pred_RF = pd.DataFrame({'ImageId' : range(1, len(y_val)+1), 'Label' : y_val})
    pred_RF.to_csv('/home/msweber/Documents/msds422/pred_PCA_RF.csv',
                   index = False)



 