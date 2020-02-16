#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:25:15 2019

@author: msweber
"""

import numpy as np
import random

hidden_layers = 5
input_shape = 13
output_shape = 8
        
layer_dict = {}
for i in range(hidden_layers):
    if i == 0:
        layer_dict.update({i : [np.random.randint(4, 32), input_shape, "relu"]})
    elif i == hidden_layers - 1:
        layer_dict.update({i : [output_shape, layer_dict.get(i-1)[0], "softmax"]})
    else:
        layer_dict.update({i : [np.random.randint(4, 32), layer_dict.get(i-1)[0], "relu"]})
        
layer_dict


l_dict = {0: [6, 13, 'relu', 0.1570171309321402], 
              1: [24, 6, 'relu', 0.46711418675751715], 
              2: [17, 24, 'relu', 0.971881697124282], 
              3: [26, 17, 'relu', 0.8970871311368472], 
              4: [8, 26, 'softmax', 0]}

a = l_dict.get(i)[0]
b = l_dict.get(i)[1]
c = l_dict.get(i)[2]


model_array = [800, 5, 7, 0.094130988664677, 0.06376377425438721,
               {0: [32, 13, 'linear', 0.07932753381542346],
                1: [17, 32, 'linear', 0.3279583855700312],
                2: [25, 17, 'linear', 0.15029129346981662],
                3: [22, 25, 'linear', 0.9827297643249093],
                4: [25, 22, 'linear', 0.3285284099098774],
                5: [16, 25, 'linear', 0.42906202787412195],
                6: [6, 16, 'linear', 0.8993373431942391],
                7: [28, 6, 'linear', 0.5794856461457433],
                8: [8, 28, 'softmax', 0]}]
model_layers = model_array[5]
model_layers.get(0)[0]

model_array[5]

for i in range(5):
    print(i)
    
random.sample(range(99), 5)



from Tkinter import *

root = Tk()
T = Text(root, height=2, width=30)
T.pack()
T.insert(END, "Just a text Widget\nin two lines\n")
mainloop()


import numpy as np
import pandas as pd
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

train = pd.DataFrame(x_train).copy()

len(train.columns)


        self.lf = {1 : 'mean_squared_error',
                   2 : 'mean_absolute_error',
                   3 : 'categorical_crossentropy', 
                   4 : 'sparse_categorical_crossentropy',
                   5 : 'kullback_leibler_divergence',
                   6 : 'poisson',
                   7 : 'categorical_hinge',
                   8 : 'hinge'}