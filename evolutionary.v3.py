#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:12:32 2019

@author: msweber
"""

#import Required Packages
import pandas as pd
import numpy as np



class evolutionary:
    def __init__(self):
        # text is the target word or phrase (letters and spaces only)
        self.text = 'Allison Turing Bus'
        # cmratio is crossover/mutation ratio
        self.cmratio = 0.8
        # tolerance is the number of cycles without improvement to stop 
        self.tolerance = 5
        # t_count is the count of arrays pulled for scoring
        self.t_count = 5
        # p is the population size
        self.p = 10
        # Start iteration count
        self.iteration = 0
        # create an array to track population scores of iterations
        self.i_score = []
        self.pop = []
        self.pop_scores = {}
        self.new_pop = []
        self.new_pop_scores = {}
       
        
        # Define Target
        self.dictt = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9,
                 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 'p':16, 'q':17,
                 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25,
                 'z':26, ' ':27}
        self.target = []
        self.l_text = self.text.lower()
        for i in list(self.l_text):
            self.target.append(self.dictt[i])
        
    def polulation(self):
        # Create random population of 1000 arrays
        # L = length of string
        self.l = len(list(self.text))
        for i in range(0, self.p):
            rand_phrase = np.random.randint(1,27,self.l)
            self.pop.append(rand_phrase)
            for s in range (0, self.l): 
                score = 0
                if rand_phrase[i] == self.target[i]:
                    score += 1
                if score == self.l:
                    print("converged on matching phrase after " + str(self.iteration) + "iterations")
                    exit
                self.pop_scores.update({s : score})
    
    def tournament (self):
        self.ts = []
        self.ts = np.randint(0, self.p, self.t_count))
        self.ts = np.random.choice(self.pop, self.t_count, replace = False)
        self.max_ts = 0
        self.max_score = 0
        for s in 