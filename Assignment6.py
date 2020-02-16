#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: Assignment 5.py
Class: MSDS422

Description:    

You will continue work on the Digit Recognition problem in Kaggle.com this week.   
As in Assignment 5, we will assess classification performance accuracy and 
processing time. Python TensorFlow should be used for Assignment 6. (If you 
experience difficulty installing TensorFlow, Python scikit-learn may be used 
as an alternative for Assignment 6.)

The Benchmark Experiment
Tested neural network structures should be explored within a benchmark 
experiment, a factorial design with at least two levels on each of two 
experimental factors (at least a 2x2 completely crossed design). But due to the 
time required to fit each neural network, we will observe only one trial for 
each cell in the design.  You will build your models on train.csv and submit 
your forecasts for test.csv to Kaggle.com, providing your name and user ID for 
each experimental trial..

An example experiment could include two values for the number of nodes per 
inner layer and two values for the number of inner layers. Various machine 
learning hyperparameter settings may be used.

Students are encouraged to work in study teams on this assignment, with the 
understanding that each student must run the code himself/herself and write an 
independent report of the experimental results. 

Original Author: msweber
Date Created: Fri Feb 14 12:09:18 2020

List of Modifications:
"""
"""
##############################################
Import required libraries
##############################################
"""
import os
import matplotlib as plt
import tensorflow as tf
import pandas as pd
import numpy as np


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

train = train = pd.read_csv("/usr/lfs/v0/msds422/train.csv")
y = train['label'].values
X_train = train.drop(['label'], axis = 1).values

test = pd.read_csv("/usr/lfs/v0/msds422/test.csv")
X_val = test.values



