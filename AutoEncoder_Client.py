#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:59:20 2019

@author: msweber
"""
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop
from keras.utils import multi_gpu_model
from keras.models import load_model
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from AutoEncoder_Builder import AE_Build
import gc

class AE_Client:
    
    def __init__(self, optimizer, activation, layers, initial_file, input_path):
        self.optimizer = optimizer
        self.activation = activation
        self.layers = layers
        self.initial_file = pd.read_csv(input_path + initial_file)
        self.path = input_path
        
        
    def Initial_Training(self, lr):
        opt = RMSprop(lr = lr)
        AE = AE_Build(opt, self.activation, self.layers, self.initial_file.shape[1])
        model = AE.create_model()
        model.summary()
        checkpoint = ModelCheckpoint(filepath='JA804A.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        data = np.asarray(self.initial_file)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config) 
        tf.global_variables_initializer
        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
            history = model.fit(data, data, 512, 4000, verbose=1, callbacks = callbacks_list)
        return history
    
    def Stepped_lr(self, step, epochs, lr_array):
        stepped_loss_record = {}
        for lr in lr_array:
            if step == 0:
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                print('xxxxxxxxxxxxxxxxx   ' + str(step) + ' : ' + str(lr) + '   xxxxxxxxxxxxxxxxxxx')
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                self.Initial_Training(lr)
                step +=1
            else:
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                print('xxxxxxxxxxxxxxxxx   ' + str(step) + ' : ' + str(lr) + '   xxxxxxxxxxxxxxxxxxx')
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                step += 1
                opt = RMSprop(lr = lr)
                AE = AE_Build(opt, self.activation, self.layers, self.initial_file.shape[1])
                model = AE.create_model()
                model.summary()
                checkpoint = ModelCheckpoint(filepath='JA804A.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')
                callbacks_list = [checkpoint]
                data = np.asarray(self.initial_file)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config) 
                with tf.Session() as sess:
#                    sess.run(tf.global_variables_initializer())
                    model.load_weights('JA804A.h5')
                    history = model.fit(data, data, 128, epochs, verbose=1, callbacks = callbacks_list)
                    stepped_loss_record.update({lr: history.history['loss']})
        pd.DataFrame.from_dict(data=stepped_loss_record, orient='index').to_csv('steppedLossRecord.csv', header=False)
        return history
    
    
    def AdamTraining(self, epochs):
        opt = self.optimizer
        AE = AE_Build(opt, self.activation, self.layers, self.initial_file.shape[1])
        model = AE.create_model()
        model.summary()
        checkpoint = ModelCheckpoint(filepath='JA804A_a.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')
        csv_logger = CSVLogger('AdamTraining.csv')
        callbacks_list = [checkpoint, csv_logger]
        data = np.asarray(self.initial_file)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config) 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.load_weights('JA804A_a.h5')
            history = model.fit(data, data, 128, epochs, verbose=1, callbacks = callbacks_list)
        return history
    
    
    def RMSTraining(self, epochs):
        opt = RMSprop(lr=0.00001, rho=0.9)
        AE = AE_Build(opt, self.activation, self.layers, self.initial_file.shape[1])
        model = AE.create_model()
        model.summary()
        checkpoint = ModelCheckpoint(filepath='JA804A_r.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')
        csv_logger = CSVLogger('RMSTraining.csv')
        callbacks_list = [checkpoint, csv_logger]
        data = np.asarray(self.initial_file)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config) 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.load_weights('JA804A_r.h5')
            history = model.fit(data, data, 128, epochs, verbose=1, callbacks = callbacks_list)
        return history
    
    
    def File_Training(self, f, data, model, callbacks_list):
        try:
            model.load_weights('JA804A_c.h5')
            history = model.fit(data, data, 128, 1000, verbose=1, callbacks = callbacks_list)
            del data
            del model
            gc.collect()
            return history
        except Exception as e:
            print(str(e) + '    in file ' + f)

    
    
    def Continued_Training(self):
        files = np.asarray(os.listdir(self.path))
        fileno = 0
        for f in files:
            data = pd.read_csv(self.path + f, low_memory = False)
            AE2 = AE_Build(self.optimizer, self.activation, self.layers, data.shape[1])
            model = AE2.create_model()
            model.summary()
            checkpoint = ModelCheckpoint(filepath='JA804A_c.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')
            csv_logger = CSVLogger('training.csv')
            es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            callbacks_list = [checkpoint, csv_logger, es]
            data = np.asarray(data)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config) 
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                history = self.File_Training(f, data, model, callbacks_list)
            del data
            del model
            gc.collect()
            fileno +=1
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print('xxxxx file ' + f + ' completed: ' + str(fileno) + ' of ' + str(len(files)) + ' xxxxx')
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        return history
    
    

        

        

