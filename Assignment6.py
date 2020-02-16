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
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import math


train = train = pd.read_csv("/usr/lfs/v0/msds422/train.csv")
test = pd.read_csv("/usr/lfs/v0/msds422/test.csv")

###############################################################################
############  Multilayer Perceptron Benchmark 2 layers x 10 nodes  ############
###############################################################################
# https://www.kaggle.com/fox10225fox/multi-layer-network-with-tensorflow


y = train['label'].values
X_train = train.drop(['label'], axis = 1).values/255
X_val = test.values/255

train_x, test_x, train_y, test_y = train_test_split(X_train, y, test_size=0.10, 
                                                    random_state=42)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

num_units = 512

# Features and Labels
features = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])

# Layers
w4 = tf.Variable(tf.random_normal([784, num_units]))
b4 = tf.Variable(tf.random_normal([num_units]))
z4 = tf.add(tf.matmul(features, w4), b4)
h4 = tf.nn.relu(z4)

w3 = tf.Variable(tf.random_normal([num_units, num_units]))
b3 = tf.Variable(tf.random_normal([num_units]))
z3 = tf.add(tf.matmul(h4, w3), b3)
h3 = tf.nn.relu(z3)

w2 = tf.Variable(tf.random_normal([num_units, num_units]))
b2 = tf.Variable(tf.random_normal([num_units]))
z2 = tf.add(tf.matmul(h3, w2), b2)
h2 = tf.nn.relu(z2)

w1 = tf.Variable(tf.random_normal([num_units, num_units]))
b1 = tf.Variable(tf.random_normal([num_units]))
z1 = tf.add(tf.matmul(h2, w1), b1)
h1 = tf.nn.relu(z1)

w0 = tf.Variable(tf.random_normal([num_units, 10]))
b0 = tf.Variable(tf.random_normal([10]))
logits = tf.add(tf.matmul(h1, w0), b0)

# Define cost and optimizer
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
predict = tf.argmax(logits, 1)
correct_prediction = tf.equal(predict, tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

def print_epoch_stats(epoch_i, sess, last_features, last_labels):
    """
    Print cost and validation accuracy of an epoch
    """
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features, labels: last_labels})
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_x, labels: test_y})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch_i,
        current_cost,
        valid_accuracy))
    
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    outout_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)
        
    return outout_batches

batch_size = 128
epochs = 20
learn_rate = 0.0001

train_batches = batches(batch_size, train_x, train_y)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch_i in range(epochs):

        # Loop over all batches
        for batch_features, batch_labels in train_batches:
            train_feed_dict = {
                features: batch_features,
                labels: batch_labels,
                learning_rate: learn_rate}
            sess.run(optimizer, feed_dict=train_feed_dict)

        # Print cost and validation accuracy of an epoch
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

    predictions = sess.run(
                        predict, 
                        feed_dict={features: test_x})
    
    
    
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
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=X_train,y=y, epochs=1000)


image_index = 11651
plt.imshow(X_val[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(X_val[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())

y_pred = model.predict(X_val)
y_val = pd.DataFrame({'ImageId' : range(1, len(y_pred)+1), 
                       'Label' : np.argmax(y_pred, axis=1)})
y_val.to_csv('keras_nn.csv', index = False)