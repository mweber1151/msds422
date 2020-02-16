#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:48:38 2019

@author: msweber
"""


from boston_preprocess import preprocess as pp
from EvModel import ev_model as ev

Data = pp()

results = Data.munge()
[input_shape, output_shape, xtrain, ytrain, xtest, ytest] = results


modelClass = ev(results)

modelClass.create_pop()
modelClass.score_pop()
