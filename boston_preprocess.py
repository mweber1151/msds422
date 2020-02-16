#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:48:38 2019

@author: msweber
"""

import numpy as np
import pandas as pd
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
from boston import ev_model as ev
from keras import Sequential
import matplotlib.pyplot as plt



class preprocess:

    def __init__(self):
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.input_shape = None
        self.output_shape = 8
    
        
    def munge(self):
        """
        ###############################################################################
        Bin Output Data and One-Hot-Encode
        ###############################################################################
        """
       
        yt = pd.DataFrame(y_train)
        yt = yt.sort_values(by=[0]).reset_index()
        
        yt1 = yt[0][51]
        yt2 = yt[0][101]
        yt3 = yt[0][152]
        yt4 = yt[0][202]
        yt5 = yt[0][253]
        yt6 = yt[0][303]
        yt7 = yt[0][354]
        
        output = []
        
        for i in range(0,len(y_train)):
            if y_train[i] < yt1:
                output.append(1)
            if y_train[i] >= yt1 and y_train[i] < yt2:
                output.append(2)
            if y_train[i] >= yt2 and y_train[i] < yt3:
                output.append(3)
            if y_train[i] >= yt3 and y_train[i] < yt4:
                output.append(4)
            if y_train[i] >= yt4 and y_train[i] < yt5:
                output.append(5)
            if y_train[i] >= yt5 and y_train[i] < yt6:
                output.append(6)
            if y_train[i] >= yt6 and y_train[i] < yt7:
                output.append(7)
            if y_train[i]>= yt7:
                output.append(8)
        
        ytrain2 = pd.get_dummies(output).values
        ytrain = np.concatenate((ytrain2, ytrain2, ytrain2))
        
        output2 = []
        
        for i in range(0,len(y_test)):
            if y_test[i] < yt1:
                output2.append(1)
            if y_test[i] >= yt1 and y_test[i] < yt2:
                output2.append(2)
            if y_test[i] >= yt2 and y_test[i] < yt3:
                output2.append(3)
            if y_test[i] >= yt3 and y_test[i] < yt4:
                output2.append(4)
            if y_test[i] >= yt4 and y_test[i] < yt5:
                output2.append(5)
            if y_test[i] >= yt5 and y_test[i] < yt6:
                output2.append(6)
            if y_test[i] >= yt6 and y_test[i] < yt7:
                output2.append(7)
            if y_test[i]>= yt7:
                output2.append(8)
        
        ytest = pd.get_dummies(output2).values
        
        """
        ###############################################################################
        Normalize Training Data
        ###############################################################################
        """
        train = pd.DataFrame(x_train).copy()
        test = pd.DataFrame(x_test).copy()
        
        self.input_shape = len(train.columns)
                  
        for i in range(self.input_shape):
            m1 = train[i][train[i].values.argmax()]
            m2 = test[i][test[i].values.argmax()]
            m0 = max(m1,m2)
            for x in range(0,len(train)):
                train[i][x] = train[i][x] / m0
            for x in range(0,len(test)):
                test[i][x] = test[i][x] / m0
                
        xtrain2 = train.values
        xtrain = np.concatenate((xtrain2, xtrain2, xtrain2))
        xtest = test.values
        
        """
        ###############################################################################
        return data to feed into models
        ###############################################################################
        """

        
        return [self.input_shape, self.output_shape, xtrain, ytrain, xtest, ytest]






'''
modelClass = ev(13,8, xtrain, ytrain, xtest, ytest)

modelClass.create_pop()
modelClass.run_models()



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pop = modelClass.create_pop()

pop
'''