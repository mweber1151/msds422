#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:42:07 2020

@author: msweber
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

class P_C_A:
    
    def __init__(self, X):
        self.X = X
        
    
    def explVar(self):
        pca = PCA().fit(self.X)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        return plt
        
    def P_C_A_(self, X, n_comp):
        pca = PCA(n_components=n_comp)
        X_reduced = pca.fit_transform(self.X)
        evr = pca.explained_variance_ratio_
        return pca.components_
        
        
        