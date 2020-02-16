#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:51:31 2019

@author: msweber
"""


# -*- coding: utf-8 -*-

"""
File: AutoEncoder_Builder.py
Project: Ascentia
Project Lead: Mike Weber

Description: Class for simple AutoEncoder

Original Author: Mike Weber
Date Created: October 29, 2019

List of Modifications:

    
    
"""


"""
##############################################
Import required libraries
##############################################
"""

from keras.models import Sequential
#from keras.utils import multi_gpu_model
from keras.layers import Dense

class AE_Build:
    
    def __init__(self, optimizer, activation, layers, input_shape):
        self.opt = optimizer
        self.act = activation
        self.layers = layers
        self.input_shape = input_shape

                 
    def create_model(self):
        input_shape = self.input_shape
        model = Sequential()
        stepWidth = round(input_shape / (self.layers * 2 - 1))
        model.add(Dense((input_shape - stepWidth), input_shape=(self.input_shape,), activation=self.act))
        for i in range(2, int(self.layers/2)):
            model.add(Dense(input_shape - (i*stepWidth), activation=self.act))
        for i in range(int(self.layers/2)+1, 1, -1):
            model.add(Dense(input_shape - (i*stepWidth), activation=self.act))
        model.add(Dense(input_shape, activation=self.act))
        model._make_predict_function()
        #model = multi_gpu_model(model, gpus=4)
        model.compile(optimizer=self.opt, loss='mean_squared_error', metrics=['acc'])
        return model
