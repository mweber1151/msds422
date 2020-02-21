# -*- coding: utf-8 -*-

"""
File: CNNBuilder.py
Project: Allison Model Framework
Project Lead: Joseph Engler

Description: Class for building convolutional neural network models

Original Author: Joseph Engler
Date Created: Tues. June 25, 2019

List of Modifications:

    
    
"""


"""
##############################################
Import required libraries
##############################################
"""
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.applications import ResNet50
from keras.optimizers import Adam
import random
import numpy as np
import tensorflow as tf
from ModelBase import BaseModel


class ConvolutionalModel(BaseModel):
    
    '''
    ###########################################################################
    Method for initializing the class scope variables
    
    Parameters:
        input_space_shape: shape of network input
        output_space_shape: shape of output
        layersDict: dictionary of layers to include in the network
        loss_function: loss function to use
        optimizer: optimizer to use
    ###########################################################################
    '''
    def __init__(self, input_shape, output_shape):
        #b.__init__(input_space_shape, output_space_shape, loss_function, optimizer)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        self.epsilon = 0.9
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.00
        self.checkpoint = ModelCheckpoint(filepath='convmodel.h5', monitor='acc', verbose=0, save_best_only=True, mode='max')
        self.graph = None
        self.optimizer = Adam(lr = 0.0001)
    
    '''
    ###########################################################################
    Method for building the model       
    
    Parameters:
    Return: 
        compiled model
    ###########################################################################
    '''
    def BuildModel(self):          
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape= self.input_shape)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(self.output_shape, activation='softmax')(x)
        model = Model(input=base_model.input, output=x)
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer= self.optimizer, loss='mean_squared_error', metrics=['accuracy'])
        model.summary()  
        self.model = model
        
        return model
        
    
    '''
    ###########################################################################
    Method for training the model      
    
    Parameters:
        batch_inputs: array of properly shaped inputs
        batch_targets: array of properly shaped target values
        batchSize: size of each batch
        epochs: Number of epochs to train for
    Return: 
        history of training
    ###########################################################################
    '''
    def Fit(self, batch_inputs, batch_labels, batchSize, epochs):
        callbacks_list = [self.checkpoint] 
        history = self.model.fit(batch_inputs, batch_labels, batch_size=batchSize, epochs=epochs, callbacks=callbacks_list, verbose=1)        
        return history.history['acc']
    
    '''
    ###########################################################################
    Method for saving the model       
    
    Parameters:
        modelName: name of model
        filePath: path to save to        
    Return: 
       
    ###########################################################################
    '''
    def Save(self, modelName, filePath):
        model_json = self.model.to_json()
        with open(filePath + '/' + modelName + '.json', 'r') as jsonFile:
            jsonFile.write(model_json)
        self.model.save_weights(filePath + '/' + modelName + '.h5')

#model testing code
if __name__ == '__main__' :
    import numpy as np
    import random
    from keras.datasets import mnist
    import cv2
    import os
    
    data = []
    labels = []
    cnt = 0
    for f in os.listdir('/usr/lfs/v0/code/AllisonModelFramework/cats/CAT_00/'):
        try:
            img = cv2.imread('/usr/lfs/v0/code/AllisonModelFramework/cats/CAT_00/' + f.replace('.cat',''))
            
            '''sizes must be square (e.g. 224 x 224)'''
            img = cv2.resize(img, (224,224))
            
            '''reshape to put the channels first'''
            data.append(np.reshape(img, (224,224,3)))
            r = random.random()
            if r > 0.5:
                labels.append([0,1])
            else:
                labels.append([1,0])
            cnt += 1
            if cnt > 2000:
                break
        except:
            pass
    data = np.asarray(data)
    labels = np.asarray(labels)
    aem = ConvolutionalModel((224,224,3),2)
    aem.BuildModel()
    aem.Fit(data, labels, 16, 200)
      
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        