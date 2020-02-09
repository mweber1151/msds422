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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime


class Grid:
    
    def __init__(self, params):
        train = pd.read_csv("/home/msweber/Documents/msds422/train.csv")
        test = pd.read_csv("/home/msweber/Documents/msds422/test.csv")
        #Extract target variable
        self.y = train['label'].values
        #Drop target variable from dataframe and make a matrix out of the remaining data
        train1 = train.drop(['label'], axis = 1)
        self.X = train1.values
        #create matrix out of dataframe
        self.X_val = test.values
        n_estimators, max_depth, max_features = params
        self.hyperparams = {'n_estimators': n_estimators, 
                            'max_depth': max_depth,
                            'max_features': max_features}
        

    def grid(self):
        gd = GridSearchCV(estimator=RandomForestClassifier(), 
                          param_grid=self.hyperparams,
                          verbose=True, 
                          cv=10, 
                          scoring='explained_variance',
                          n_jobs=-1
                          )
        gd.fit(self.X, self.y)
        print(gd.best_score_)
        print(gd.best_estimator_)
        return gd.best_score_, gd.best_estimator_
        
    
if __name__ == '__main__':
    n_estimators = [50, 100, 500, 1000]
    max_depth = [5, 10, None ]
    max_features = [5, 10, 15, 20, 30, 50]
    ##########################################
    #####          For Testing           #####
    ##########################################
    #n_estimators = [5, 10, 20]
    #max_depth = [2, 4, None]
    #max_features = [5, 10, 15]
    
    params = (n_estimators, max_depth, max_features)
        
    grid = Grid(params)
    start_time = datetime.now()
    print('Start time = ' + str(datetime.now()))
    best_score, best_est = grid.grid()            
    elapsed_time = datetime.now() - start_time
    print('Elapsed time = ' + str(elapsed_time))