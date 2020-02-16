#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:28:56 2019

@author: msweber
"""

from keras import optimizers
from keras import models
from keras import layers
from keras import Sequential
import matplotlib.pyplot as plt

hidden_layers = 5
input_shape = 13
output_shape = 8
        
layer_dict = {}
for i in range(hidden_layers):
    if i == 0:
        layer_dict.update({i : [np.random.randint(4, 32), input_shape, "relu", np.random.uniform(0,1)]})
    elif i == hidden_layers - 1:
        layer_dict.update({i : [output_shape, layer_dict.get(i-1)[0], "softmax", 0]})
    else:
        layer_dict.update({i : [np.random.randint(4, 32), layer_dict.get(i-1)[0], "relu", np.random.uniform(0,1)]})
 

               
model = Sequential()
for i in range(hidden_layers):
    a = layer_dict.get(i)[0]
    b = layer_dict.get(i)[1]
    c = layer_dict.get(i)[2]
    d = layer_dict.get(i)[3] 
    
    if i == hidden_layers - 1:
        model.add(layers.Dense(a, input_dim = b, activation = c))
    else:
        model.add(layers.Dense(a , input_dim = b, activation = c))
        model.add(layers.Dropout(d))
optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
#optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics = ['accuracy'])
model.summary()

history = model.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=500, batch_size=64)

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


'''
model.add(layers.Dense(20, input_dim=13, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(30, input_dim=20, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(21, input_dim=30, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(12, input_dim=21, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(8, input_dim=10, activation='softmax'))
'''