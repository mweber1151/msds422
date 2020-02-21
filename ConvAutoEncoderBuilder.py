# -*- coding: utf-8 -*-

"""
File: AutoEncoderBuilder.py
Project: Allison Model Framework
Project Lead: Joseph Engler

Description: Class for building autoencoder models

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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten,Reshape, UpSampling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import random
import numpy as np
import tensorflow as tf
from ModelBase import BaseModel


class ConvAutoEncoderModel(BaseModel):
    
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
    def __init__(self, channels=1, width=200, height=200):
        #b.__init__(input_space_shape, output_space_shape, loss_function, optimizer)
        self.input_shape = (channels, width, height)
        self.output_shape = self.input_shape
        self.upsampleFilters = int(width/4.0)       
        self.channels = channels
        self.model = None
        self.epsilon = 0.9
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.00
        self.checkpoint = ModelCheckpoint(filepath='convautoencodermodel.h5', monitor='acc', verbose=0, save_best_only=True, mode='max')
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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config) 
        with tf.Session() as sess:                  
            
            input_img = Input(shape=self.input_shape)
            # Encoder
            x = Conv2D(8,(3,3),
                       activation='relu',
                       padding='same',
                       data_format='channels_first')(input_img)
            x = Conv2D(8,(3,3),
                       activation='relu',
                       padding='same',
                       data_format='channels_first')(x)
            x = MaxPooling2D((2,2),
                             padding='same',
                             data_format='channels_first')(x) 
            x = Conv2D(16,(3,3),
                       activation='relu',
                       padding='same',
                       data_format='channels_first')(x)
            x = Conv2D(16,(3,3),
                       activation='relu',
                       padding='same',
                       data_format='channels_first')(x)
            x = MaxPooling2D((2,2),
                             padding='same',
                             data_format='channels_first')(x) 
            x = Flatten()(x)
            code = Dense(256)(x)
            # Decoder
            x = Dense(16*self.upsampleFilters*self.upsampleFilters)(code)
            x = Reshape((16,self.upsampleFilters,self.upsampleFilters))(x)
            x = UpSampling2D((2, 2),
                             data_format='channels_first')(x)
            x = Conv2D(16, (3, 3),
                       activation='relu',
                       padding='same',
                       data_format='channels_first')(x)
            x = Conv2D(16, (3, 3),
                       activation='relu',
                       padding='same',
                       data_format='channels_first')(x)
            x = UpSampling2D((2, 2),
                             data_format='channels_first')(x)  
            x = Conv2D(8, (3, 3),
                       activation='relu',
                       padding='same',
                       data_format='channels_first')(x)
            decoded = Conv2D(self.channels, (3, 3),
                       activation='relu',
                       padding='same',
                       data_format='channels_first')(x)
        
            model = Model(input_img, decoded)
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
    def Fit(self, batch_inputs, batchSize, epochs):
        callbacks_list = [self.checkpoint] 
        history = self.model.fit(batch_inputs, batch_inputs, batch_size=batchSize, epochs=epochs, callbacks=callbacks_list, verbose=1)        
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
    
'''
#model testing code
if __name__ == '__main__' :
    import numpy as np
    import random
    from keras.datasets import mnist
    import cv2
    import os
    
    data = []
    cnt = 0
    for f in os.listdir('/usr/lfs/v0/code/AllisonModelFramework/cats/CAT_00/'):
        try:
            img = cv2.imread('/usr/lfs/v0/code/AllisonModelFramework/cats/CAT_00/' + f.replace('.cat',''))
            
            '''sizes must be square (e.g. 200 x 200)'''
            img = cv2.resize(img, (400,400))
            
            '''reshape to put the channels first'''
            data.append(np.reshape(img, (3, 400,400)))
            cnt += 1
            if cnt > 2000:
                break
        except:
            pass
    data = np.asarray(data)
    
    aem = ConvAutoEncoderModel(3,400,400)
    aem.BuildModel()
    aem.Fit(data, 128, 200)
      
'''      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        