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


class MNIST:
    
    def __init__(self):
        train = pd.read_csv("/home/msweber/Documents/MNIST/train.csv")
        test = pd.read_csv("/home/msweber/Documents/MNIST/test.csv")
        #Extract target variable
        self.y = train['label'].values
        #Drop target variable from dataframe and make a matrix out of the remaining data
        train1 = train.drop(['label'], axis = 1)
        self.X = train1.as_matrix()
        #create matrix out of dataframe
        self.X_val = test.as_matrix()
        self.RFC = None
        self.valAcc = None

    def tts(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                        test_size=0.2)
        data =  (X_train, X_test, y_train, y_test)
        return data

    def randForClass(self, data, n_est):
        X_train, X_test, y_train, y_test = data
        start_time = datetime.now()
        print("Start Time: " + str(start_time))
        RFC = RandomForestClassifier(n_estimators=n_est)
        RFC.fit(X_train, y_train)
        y_pred = RFC.predict(X_test)
        end_time = datetime.now()
        print("End Time: " + str(end_time))
        elapsed_time = end_time - start_time
        print("Elapsed Time: " + str(elapsed_time))
        valAcc = str(accuracy_score(y_test, y_pred))
        print("Validation Accuracy: " + valAcc)
        return RFC, valAcc, elapsed_time
    
    def cv(self, mult, n_est):
        RF = None
        VA = 0
        ET = 0
        it = 0
        data = self.tts()
        for n in range(1, mult):
            RFC, valAcc, elapsed_time = self.randForClass(data, n_est)
            if valAcc > VA:
                RF = RFC
                VA = valAcc
                ET = elapsed_time
            it += 1
            print("Iteration #" + str(it) + " completed")   
        return RF, VA, ET
    
mnist = MNIST()
RF, VA, ET = mnist.cv(20, 500)
            
