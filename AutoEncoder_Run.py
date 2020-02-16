#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:16:18 2019

@author: msweber
"""

from AutoEncoder_Builder import AE_Build
from AutoEncoder_Client import AE_Client
from keras.optimizers import RMSprop, Adam
import os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import tensorflow as tf
import gc

AE = AE_Client(Adam(lr=0.0001), 'relu', 17, 'z_JA804A_1.csv', '/rfs/public/Code/Ascentia/inputData/')
#AE.Stepped_lr(1, 2000 , [0.0001, 0.00001, 0.000007, 0.000004, 0.000002, 0.000001, 0.0000001])
#AE.Stepped_lr(1, 3000 , [0.0000005, 0.0000001])
#AE.Continued_Training()
#AE.AdamTraining(1)
#AE.RMSTraining(5000)

path = '/rfs/public/Code/Ascentia/inputData/'
files = np.asarray(os.listdir(path))
fileno = 0
training_log = {}
for f in files:
    data = pd.read_csv(path + f, low_memory = False)
    AE2 = AE_Build(Adam(lr=0.000001), 'relu', 17, data.shape[1])
    model = AE2.create_model()
    model.summary()
    checkpoint = ModelCheckpoint(filepath='JA804A_c2.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')
    csv_logger = CSVLogger('training.csv')
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
    callbacks_list = [checkpoint, csv_logger, es]
    data = np.asarray(data)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_weights('JA804A_c2.h5')
        history = model.fit(data, data, 128, 500, verbose=1, callbacks = callbacks_list)
    training_log.update({f: history.history})
    df = pd.DataFrame.from_dict(training_log, orient="index")
    df.to_csv("continued_training.csv")
    del data
    del model
    gc.collect()
    fileno +=1
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('xxxxx file ' + f + ' completed: ' + str(fileno) + ' of ' + str(len(files)) + ' xxxxx')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
