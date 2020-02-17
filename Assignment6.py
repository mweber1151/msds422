#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: Assignment 6.py
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
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import math
from datetime import datetime



train = train = pd.read_csv("~/Documents/msds422/train.csv")
test = pd.read_csv("~/Documents/msds422/test.csv")




###############################################################################
###########  Multilayer Perceptron Benchmark 2 layers x 500 nodes  ############
###############################################################################
# 
'''

y = train['label'].values
X_train = train.drop(['label'], axis = 1).values/255
X_val = test.values/255

train_x, test_x, train_y, test_y = train_test_split(X_train, y, test_size=0.10, 
                                                    random_state=42)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

n_inputs = 784
n_hidden1 = 500
n_hidden2 = 500
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32,shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation='relu')
    logits = neuron_layer(hidden2, n_outputs, "outputs")
    
with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=train_y,
                        logits=neuron_layer(hidden2, n_outputs, "outputs"),
                        name=None)
    loss = tf.reduce_mean(xentropy, name="loss")
    
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer =tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    print(logits)
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer
saver = tf.train.Saver()

n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(train_x.num_examples / batch_size):
            X_batch, y_batch = train_x.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: test_x, y: test_y})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")
    
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = X_val
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z)

'''    
###############################################################################
############        2 Layer Dense Neural Network with Keras        ############
###############################################################################

y = train['label'].values
X_train = train.drop(['label'], axis = 1).values
X_train[3].shape

X_val = test.values

# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_val /= 255
print('X_train shape:', X_train.shape)
print('Number of images in X_train', X_train.shape[0])
print('Number of images in X_val', X_val.shape[0])


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Dense(250, input_dim=784, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(250, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))
model._make_predict_function()
model.summary()
checkpoint = ModelCheckpoint(filepath='tf_dnn.h5', monitor='val_accuracy', 
                             verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
tf.global_variables_initializer
start_time = datetime.now()
print("Start Time: " + str(start_time))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    history = model.fit(x=X_train,y=y, epochs=50, validation_split=0.20,
                        verbose=1, callbacks = callbacks_list)
end_time = datetime.now()
print("End Time: " + str(end_time))
DNN_time = end_time - start_time
print("2 Layer Dense Neural Network with Keras Elapsed Time: " + str(DNN_time))

y_pred = model.predict(X_val)
y_val = pd.DataFrame({'ImageId' : range(1, len(y_pred)+1), 
                       'Label' : np.argmax(y_pred, axis=1)})
y_val.to_csv('tf_dnn.csv', index = False)
 

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('tf_dnn model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('tf_dnn model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

tf_dnn_accuracy = np.max(history.history['accuracy'])
tf_dnn_val_acc = np.max(history.history['val_accuracy'])
###############################################################################
############  2 Layer Dense Neural Network with Keras (Tapered)    ############
###############################################################################

y = train['label'].values
X_train = train.drop(['label'], axis = 1).values
X_train[3].shape

X_val = test.values

# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_val /= 255
print('X_train shape:', X_train.shape)
print('Number of images in X_train', X_train.shape[0])
print('Number of images in X_val', X_val.shape[0])


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Dense(250, input_dim=784, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(100, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))
model._make_predict_function()
model.summary()
checkpoint = ModelCheckpoint(filepath='tf_dnn_tapered.h5', monitor='val_accuracy', 
                             verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
tf.global_variables_initializer
start_time = datetime.now()
print("Start Time: " + str(start_time))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    history = model.fit(x=X_train,y=y, epochs=50, validation_split=0.20,
                        verbose=1, callbacks = callbacks_list)
end_time = datetime.now()
print("End Time: " + str(end_time))
DNN_Tapered_time = end_time - start_time
print("2 Layer Dense Neural Network with Keras (Tapered) Elapsed Time: " + str(DNN_Tapered_time))

y_pred = model.predict(X_val)
y_val = pd.DataFrame({'ImageId' : range(1, len(y_pred)+1), 
                       'Label' : np.argmax(y_pred, axis=1)})
y_val.to_csv('tf_dnn_tapered.csv', index = False)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('tf_dnn_tapered model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('tf_dnn_tapered model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

tf_dnn_taper_accuracy = np.max(history.history['accuracy'])
tf_dnn_taper_val_acc = np.max(history.history['val_accuracy'])
###############################################################################
############        5 Layer Dense Neural Network with Keras        ############
###############################################################################

y = train['label'].values
X_train = train.drop(['label'], axis = 1).values
X_train[3].shape

X_val = test.values

# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_val /= 255
print('X_train shape:', X_train.shape)
print('Number of images in X_train', X_train.shape[0])
print('Number of images in X_val', X_val.shape[0])


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Dense(250, input_dim=784, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(250, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(250, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(250, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(250, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))
model.summary()
checkpoint = ModelCheckpoint(filepath='tf_5L_dnn.h5', monitor='val_accuracy', 
                             verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
tf.global_variables_initializer
start_time = datetime.now()
print("Start Time: " + str(start_time))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    history = model.fit(x=X_train,y=y, epochs=50, validation_split=0.20,
                        verbose=1, callbacks = callbacks_list)
end_time = datetime.now()
print("End Time: " + str(end_time))
DNN_5L_time = end_time - start_time
print("5 Layer Dense Neural Network with Keras Elapsed Time: " + str(DNN_5L_time))

y_pred = model.predict(X_val)
y_val = pd.DataFrame({'ImageId' : range(1, len(y_pred)+1), 
                       'Label' : np.argmax(y_pred, axis=1)})
y_val.to_csv('tf_5L_dnn.csv', index = False)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('tf_5L_dnn model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('tf_5L_dnn model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

tf_5L_dnn_accuracy = np.max(history.history['accuracy'])
tf_5L_dnn_val_acc = np.max(history.history['val_accuracy'])
###############################################################################
############  5 Layer Dense Neural Network with Keras (Tapered)    ############
###############################################################################

y = train['label'].values
X_train = train.drop(['label'], axis = 1).values
X_train[3].shape

X_val = test.values

# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_val /= 255
print('X_train shape:', X_train.shape)
print('Number of images in X_train', X_train.shape[0])
print('Number of images in X_val', X_val.shape[0])


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Dense(250, input_dim=784, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(200, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(150, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(100, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(50, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))
model.summary()
checkpoint = ModelCheckpoint(filepath='tf_5L_dnn_tapered.h5', monitor='val_accuracy', 
                             verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
tf.global_variables_initializer
start_time = datetime.now()
print("Start Time: " + str(start_time))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    history = model.fit(x=X_train,y=y, epochs=50, validation_split=0.20,
                        verbose=1, callbacks = callbacks_list)
end_time = datetime.now()
print("End Time: " + str(end_time))
DNN_5L_Tapered_time = end_time - start_time
print("5 Layer Dense Neural Network with Keras (Tapered) Elapsed Time: " + str(DNN_5L_Tapered_time))

y_pred = model.predict(X_val)
y_val = pd.DataFrame({'ImageId' : range(1, len(y_pred)+1), 
                       'Label' : np.argmax(y_pred, axis=1)})
y_val.to_csv('tf_5L_dnn_tapered.csv', index = False)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('tf_dnn_tapered model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('tf_dnn_tapered model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


tf_5L_taper_dnn_accuracy = np.max(history.history['accuracy'])
tf_5L_taper_dnn_val_acc = np.max(history.history['val_accuracy'])
'''

###############################################################################
############          2D Convolutional Network with Keras          ############
###############################################################################

y = train['label'].values
X_train = train.drop(['label'], axis = 1).values
X_train = X_train.reshape(42000, 28, 28)
X_train[3].shape


X_val = test.values
X_val = X_val.reshape(28000, 28, 28)


image_index = np.random.random_integers(0,42000)
print(y[image_index]) # The label is 8
plt.imshow(X_train[image_index], cmap='Greys')


# Reshaping the array to 4-dims so that it can work with the Keras API
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_val /= 255
print('X_train shape:', X_train.shape)
print('Number of images in X_train', X_train.shape[0])
print('Number of images in X_val', X_val.shape[0])


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))
model.summary()
checkpoint = ModelCheckpoint(filepath='keras_nn.csv.h5', monitor='val_accuracy', 
                             verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
tf.global_variables_initializer
start_time = datetime.now()
print("Start Time: " + str(start_time))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    history = model.fit(x=X_train,y=y, epochs=100, validation_split=0.20,
                        verbose=1, callbacks = callbacks_list)
end_time = datetime.now()
print("End Time: " + str(end_time))
CNN_time = end_time - start_time
print("2D Convolutional Neural Network Elapsed Time: " + str(CNN_time))

y_pred = model.predict(X_val)
y_val = pd.DataFrame({'ImageId' : range(1, len(y_pred)+1), 
                       'Label' : np.argmax(y_pred, axis=1)})
y_val.to_csv('keras_nn.csv', index = False)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('keras cnn model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('keras cnn model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



###############################################################################
############       2D 5L Convolutional Network with Keras          ############
###############################################################################

y = train['label'].values
X_train = train.drop(['label'], axis = 1).values
X_train = X_train.reshape(42000, 28, 28)
X_train[3].shape


X_val = test.values
X_val = X_val.reshape(28000, 28, 28)


image_index = np.random.random_integers(0,42000)
print(y[image_index]) # The label is 8
plt.imshow(X_train[image_index], cmap='Greys')


# Reshaping the array to 4-dims so that it can work with the Keras API
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_val /= 255
print('X_train shape:', X_train.shape)
print('Number of images in X_train', X_train.shape[0])
print('Number of images in X_val', X_val.shape[0])


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(650, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(450, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(300, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(150, activation=tf.nn.relu))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))
model.summary()
checkpoint = ModelCheckpoint(filepath='keras_5Lnn.csv.h5', monitor='val_accuracy', 
                             verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
tf.global_variables_initializer
start_time = datetime.now()
print("Start Time: " + str(start_time))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    history = model.fit(x=X_train,y=y, epochs=100, validation_split=0.20,
                        verbose=1, callbacks = callbacks_list)
end_time = datetime.now()
print("End Time: " + str(end_time))
CNN_5L_time = end_time - start_time
print("2D 5L Convolutional Neural Network Elapsed Time: " + str(CNN_5L_time))

y_pred = model.predict(X_val)
y_val = pd.DataFrame({'ImageId' : range(1, len(y_pred)+1), 
                       'Label' : np.argmax(y_pred, axis=1)})
y_val.to_csv('keras_5Lnn.csv', index = False)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('keras 5L cnn model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('keras 5Lcnn model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''
###############################################################################
############                Plot Image for Testing                 ############
###############################################################################



image_index = 11651
plt.imshow(X_val[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(X_val[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())