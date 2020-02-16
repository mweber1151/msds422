# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:01:50 2019

@author: jjengler
"""


from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.optimizers import RMSprop, Adam
import keras.losses as losses
from keras.models import load_model, model_from_json
import random
import numpy as np
import tensorflow as tf
import argparse
import pickle

try:
    K.clear_session()
    parser = argparse.ArgumentParser()
    parser.add_argument('fileName', nargs='*', default='modelData0.pickel')
    args = parser.parse_args()
    filename = args.fileName
    try:
        print(filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except:
        filename = args.fileName[0]
        print(filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    inputData, outputData, inputVal, outputVal, batch_size, epochs, modelName, optimizer, loss = data
    
    #model = Sequential()
    json_file = open(modelName.replace('.h5','.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #model = load_model(modelName)
    model.load_weights(modelName)
    model._make_predict_function()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = model.fit(inputData, outputData, validation_data=(inputVal, outputVal),
                                 epochs=epochs, verbose = 1, batch_size=batch_size)
    print(history.history['val_acc'][len(history.history['val_acc'])-1])
    with open(filename.replace('modelData','history').replace('.pickel','.scr'), 'w') as f:
        f.write(str(history.history['val_acc'][len(history.history['val_acc'])-1]))
except:
    with open(filename.replace('modelData','history').replace('.pickel','.scr'), 'w') as f:
        f.write('')