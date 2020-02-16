#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:24:55 2019

@author: msweber
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:39:37 2019

@author: msweber
"""

from keras.datasets import boston_housing
from keras.models import Model
from keras import layers
from keras import optimizers
import numpy as np
import pandas as pd
import random
from keras import Sequential
import pickle
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from multiprocessing.dummy import Pool as ThreadPool


def get_session():
    gpu_options = tf.GPUOptions(allow_growth = True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())


class ev_model:
    def __init__(self,results):
        self.input_shape = results[0]
        self.output_shape = results[1]
        self.xtrain = results[2]
        self.train_size = len(self.xtrain)
        self.xtest = results[3]
        self.ytrain = results[4]
        self.ytest = results[5]
        print(self.input_shape)
        print(self.output_shape)
        print(len(self.xtrain))
        print(len(self.xtest))
        print(len(self.ytrain))
        print(len(self.ytest))
        self.lr = 0.01
        self.decay = 0.1
        self.pop = {}
        self.universe = {}
        self.pop_size = 0
        self.pop_t = {}
        self.pop_t_size = 0
        self.pop_scores = []
        self.u_scores = {}
        self.ex_universe = {}
        self.ev_cycle = 0
        self.ts = 5
        self.mr = 0.2
        self.epochs = 25
        self.population = 1
        self.lf = {1 : 'mean_squared_error',
                   2 : 'mean_absolute_error',
                   3 : 'categorical_crossentropy', 
                   4 : 'sparse_categorical_crossentropy',
                   5 : 'kullback_leibler_divergence',
                   6 : 'poisson',
                   7 : 'categorical_hinge',
                   8 : 'hinge'}
        self.act = {1 : 'softmax',
                    2 : 'elu',
                    3 : 'selu', 
                    4 : 'softplus',
                    5 : 'softsign',
                    6 : 'softmax',
                    7 : 'relu',
                    8 : 'tanh',
                    9 : 'sigmoid',
                    10 : 'hard_sigmoid',
                    11 : 'linear'}
        self.op = {1 : 'SGD',
                   2 : 'RMSProp',
                   3 : 'Adagrad',
                   4 : 'Adadelta',
                   5 : 'Adam', 
                   6 : 'Adamax', 
                   7 : 'Nadam'}

        
    def build_array(self):
        #[batch size, loss function, optimizer, learning rate, decay, dict of layers]
        batch = random.randint(1,self.train_size)
        loss = random.randint(1,len(self.lf))
        opt = random.randint(1, len(self.op))
        learn = random.uniform(0.000001, .1)
        decay = random.uniform(0.0, .2)
        layers = random.randint(2, 9)
        activation = self.act.get(random.randint(1,len(self.act)))
        layer_dict = {}
        for i in range(layers):
            if i == 0:
                layer_dict.update({i : [random.randint(4, 32), self.input_shape, 
                                        activation, random.uniform(0,0.2)]})
            elif i == layers - 1:
                layer_dict.update({i : [self.output_shape, layer_dict.get(i-1)[0], "softmax", 0]})
            else:
                layer_dict.update({i : [random.randint(4, 32), layer_dict.get(i-1)[0],
                                        activation, random.uniform(0,0.2)]})
        arr = [batch, loss, opt, learn, decay, layer_dict]
        return arr
    
    
    def create_pop(self):
        while self.pop_size < self.population:
            new_array = self.build_array()
            self.pop.update({self.pop_size : new_array})
            self.pop_size += 1

                
    def parallel(self, threads = 12):
        pool = ThreadPool(threads)
        results = pool.map(self.pool_run, self.pop.values())
        pool.close()
        pool.join()
        return results
    
    
    def pool_run(self, model_array):
        model_layers = model_array[5]
        model = Sequential()
        for i in range(len(model_layers)):                
            if i == len(model_layers)-1:
                model.add(layers.Dense(model_layers.get(i)[0], input_dim = model_layers.get(i)[1], activation = model_layers.get(i)[2]))
            else:
                model.add(layers.Dense(model_layers.get(i)[0], input_dim = model_layers.get(i)[1], activation = model_layers.get(i)[2]))
                model.add(layers.Dropout(model_layers.get(i)[3]))
        optm = {1 : 'optimizers.SGD(lr=model_array[3], momentum=0.0, decay=model_array[4], nesterov=False)',
                2 : 'optimizers.RMSprop(lr=model_array[3], rho=0.9, epsilon=None, decay=model_array[4])',
                3 : 'optimizers.Adagrad(lr=model_array[3], epsilon=None, decay=model_array[4])',
                4 : 'optimizers.Adadelta(lr=model_array[3], rho=0.95, epsilon=None, decay=model_array[4])',
                5 : 'optimizers.Adam(lr=model_array[3], beta_1=0.9, beta_2=0.999, epsilon=None, decay=model_array[4], amsgrad=False)',
                6 : 'optimizers.Adamax(lr=model_array[3], beta_1=0.9, beta_2=0.999, epsilon=None, decay=model_array[4])',
                7 : 'optimizers.Nadam(lr=model_array[3], beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=model_array[4])'}
        eval(optm[model_array[2]])
        model.compile(loss=self.lf[1], optimizer=self.op[model_array[2]], metrics = ['accuracy'])
        print(model_array)
        model.summary()
        history = model.fit(self.xtrain, self.ytrain, validation_data=(self.xtest, self.ytest),
                                 epochs=self.epochs, verbose = 0, batch_size=model_array[0])
        iter_scores.update({i : history.history['val_acc']})
        score.append(iter_scores.get(i)[self.epochs -1])
        results = print(score)
        return results
        
        
    
    def run_models(self):
        score = []
        for i in range(self.pop_size):
            self.pop_d.update({i:self.pop[i]})
            model_array = self.pop[i]
            model = self.build_model(model_array)
            history = self.model.fit(self.xtrain, self.ytrain, validation_data=(self.xtest, self.ytest),
                                     epochs=self.epochs, verbose = 0, batch_size=model_array[0])
            iter_scores.update({i : history.history['val_acc']})
            score.append(iter_scores.get(i)[self.epochs -1])
        self.pop_scores.append(sum(score))
        if self.ev_cycle >= 5:
            if self.pop_scores[self.ev_cycle] <= self.pop_scores[self.ev_cycle - 5] * 1.025:
                print("the model has ceased significant improvement after " + self.ev_cycle + " evolutionary cycles")
                pop = np.asarray(self.pop)
                idx = pop[pop.argsort()[-2:]]
                best = self.pop[idx[0]].copy()
                best_model = Sequential()
                model_layers = best_model[5]
                for i in range(len(model_layers)):                
                    if i == len(model_layers):
                        best_model.add(layers.Dense(model_layers.get(i)[0], input_dim = model_layers.get(i)[1], activation = model_layers.get(i)[2]))
                    else:
                        best_model.add(layers.Dense(model_layers.get(i)[0], input_dim = model_layers.get(i)[1], activation = model_layers.get(i)[2]))
                        best_model.add(layers.Dropout(model_layers.get(i)[3]))
                optm = {1 : 'optimizers.SGD(lr=best_model[3], momentum=0.0, decay=best_model[4], nesterov=False)',
                        2 : 'optimizers.RMSprop(lr=best_model[3], rho=0.9, epsilon=None, decay=best_model[4])',
                        3 : 'optimizers.Adagrad(lr=best_model[3], epsilon=None, decay=best_model[4])',
                        4 : 'optimizers.Adadelta(lr=best_model[3], rho=0.95, epsilon=None, decay=best_model[4])',
                        5 : 'optimizers.Adam(lr=best_model[3], beta_1=0.9, beta_2=0.999, epsilon=None, decay=best_model[4], amsgrad=False)',
                        6 : 'optimizers.Adamax(lr=best_model[3], beta_1=0.9, beta_2=0.999, epsilon=None, decay=best_model[4])',
                        7 : 'optimizers.Nadam(lr=best_model[3], beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=best_model[4])'}
                eval(optm[best_model[2]])
                best_model.compile(loss=self.lf[1], optimizer=self.op[best_model[2]], metrics = ['accuracy'])
                best_model.summary()
                self.best_model = best_model
                result_model = best_model.fit(self.xtrain, self.ytrain, validation_data=(self.xtest, self.ytest), epochs=100, batch_size=best_model[0])
                result_model.save("result_model.h5")
                return result_model
                print(self.pop_scores)
        if self.ev_cycle == 0:
            print(self.pop_scores)
            self.evolution()
        else:
            print(self.pop_scores)
            self.reset_population()
                
            
    def tourney_select(self):
        ts_choice = random.sample(range(99), k = self.ts)
        arr = np.asarray(ts_choice)
        idx = arr[arr.argsort()[-2:]]
        self.t1 = self.pop[idx[0]].copy()
        self.t2 = self.pop[idx[1]].copy()
        
    def evolution(self):
        while self.pop_size < self.population:
            self.tourney_select()
            method = random.uniform(0,1)
            if method <= self.mr:
                #mutate
                m = self.t1.copy()
                element = random.randint(0, 5)
                if element <= 4:
                    elem = {0 : "random.randint(1,self.train_size)",
                            1 : "random.randint(1,len(self.lf))",
                            2 : "random.randint(1, len(self.op))",
                            3 : "random.uniform(0.000001, .1)",
                            4 : "random.uniform(0.0, .2)"}
                    m[element] = eval(elem[element])
                else:
                    layer_dict = {}
                    layers = random.randint(2, 9)
                    for i in range(layers):
                        if i == 0:
                            layer_dict.update({i : [random.randint(4, 32), self.input_shape, 
                                                    activation, random.uniform(0,0.2)]})
                        elif i == layers - 1:
                            layer_dict.update({i : [self.output_shape, layer_dict.get(i-1)[0], "softmax", 0]})
                        else:
                            layer_dict.update({i : [random.randint(4, 32), layer_dict.get(i-1)[0],
                                                    activation, random.uniform(0,0.2)]})
                m[5] = layer_dict.copy()
                self.pop_t.append(m)
                self.pop_t_size += 1
            else:
                #crossover
                m0 = self.t1.copy()
                m1 = self.t2.copy()
                m2 = np.concatenate((m0[0:element], m1[element:]))
                m3 = np.concatenate((m1[0:element], m0[element:]))
                self.pop_t.append(m2)
                self.pop_t_size += 1
                if pop_t_size <= self.population:
                    self.pop_t.append(m3)
                    self.pop_t_size += 1
        else:
            self.run_models()

                
    def reset_population(self):
        self.ev_cycle = self.ev_cycle + 1
        self.pop = self.pop_t.copy()
        self.pop_d = dict(enumerate(self.pop))
        self.pop_t = []
        self.pop_t_size = 0
        self.evolution()        
        
    
    def build_model(self, model_array):
        model_layers = model_array[5]
        model = Sequential()
        for i in range(len(model_layers)):                
            if i == len(model_layers):
                model.add(layers.Dense(model_layers.get(i)[0], input_dim = model_layers.get(i)[1], activation = model_layers.get(i)[2]))
            else:
                model.add(layers.Dense(model_layers.get(i)[0], input_dim = model_layers.get(i)[1], activation = model_layers.get(i)[2]))
                model.add(layers.Dropout(model_layers.get(i)[3]))
        optm = {1 : 'optimizers.SGD(lr=model_array[3], momentum=0.0, decay=model_array[4], nesterov=False)',
                2 : 'optimizers.RMSprop(lr=model_array[3], rho=0.9, epsilon=None, decay=model_array[4])',
                3 : 'optimizers.Adagrad(lr=model_array[3], epsilon=None, decay=model_array[4])',
                4 : 'optimizers.Adadelta(lr=model_array[3], rho=0.95, epsilon=None, decay=model_array[4])',
                5 : 'optimizers.Adam(lr=model_array[3], beta_1=0.9, beta_2=0.999, epsilon=None, decay=model_array[4], amsgrad=False)',
                6 : 'optimizers.Adamax(lr=model_array[3], beta_1=0.9, beta_2=0.999, epsilon=None, decay=model_array[4])',
                7 : 'optimizers.Nadam(lr=model_array[3], beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=model_array[4])'}
        eval(optm[model_array[2]])
        model.compile(loss=self.lf[1], optimizer=self.op[model_array[2]], metrics = ['accuracy'])
        model.summary()
        self.model = model
        #history = model.fit(self.xtrain, self.ytrain, validation_data=(self.xtest, self.ytest), epochs=100, batch_size=model_array[0])
        #return history

               
        


'''        
        self.lr = 0.01
        self.decay = 0.1
        self.loss ='categorical_crossentropy'
        self.epochs = 25
        self.batch_size = 64
        self.hidden_layers = random.randint(3, 10)
        layer_dict = {}
        for i in range(self.hidden_layers):
            if i == 0:
                layer_dict.update({i : [random.randint(4, 32), self.input_shape, "relu", random.uniform(0,1)]})
            elif i == self.hidden_layers - 1:
                layer_dict.update({i : [self.output_shape, layer_dict.get(i-1)[0], "softmax", 0]})
            else:
                layer_dict.update({i : [random.randint(4, 32), layer_dict.get(i-1)[0], "relu", random.uniform(0,1)]})
        self.layer_dict = layer_dict
   '''     
        