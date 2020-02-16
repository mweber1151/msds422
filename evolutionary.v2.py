#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:45:26 2019

@author: msweber
"""
#import Required Packages
import pandas as pd
import numpy as np

class evolutionary:
    
    def __init__(self):
        # Hyper Parameters
        self.testing = {}
        # text is the target word or phrase (letters and spaces only)
        self.text = 'Allison Turing Bus'
        # cmratio is crossover/mutation ratio
        self.cmratio = 0.8
        # tolerance is the number of cycles without improvement to stop 
        self.tolerance = 10
        # t_count is the count of arrays pulled for scoring
        self.t_count = 5
        # p is the population size
        self.p = 10
        # Start iteration count
        self.iteration = 0
        # create a dict to track population scores of iterations
        self.i_score = {}
       
        
        # Define Target
        self.dictt = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9,
                 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 'p':16, 'q':17,
                 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25,
                 'z':26, ' ':27}
        self.target = []
        self.l_text = self.text.lower()
        for i in list(self.l_text):
            self.target.append(self.dictt[i])
                
    def first_pop(self):    
        # Create random population of 1000 arrays
        # L = length of string
        self.l = len(list(self.text))
        for i in range(self.p):
            rand_phrase = np.random.randint(1,27,self.l)
            
        
        
        
        
        self.pop_t = {}
        self.pop_score = {}
        self.pop_t_score = {}
        self.pop_t1 = {}
        self.pop_t1_score = {}
        for x in range(0,self.p):
            rand_phrase = np.random.randint(1,27,self.l)
            self.pop_t.update({x : rand_phrase})
            for i in range(0,self.l):
                score = 0
                if rand_phrase[i] == self.target[i]:
                    score = score + 1
            if score == self.l:
                print("converged on matching phrase after " + str(self.iteration) + "iterations")
                exit 
            self.pop_score.update({x : score})
        
        # set q (pop_t1 size) to 0
        self.q = 0
    
   
    # generate total score
    def fitness_function(self):
        total_score = sum(self.pop_t_score)
        self.i_score.update({self.iteration : total_score})
        
    
    # randomly select arrays for tournament scoring
    def tournament(self):
        # create a blank tournament selection array and dict
        self.ts = []
        selections = {}
        self.pop_ts = {} 
        t_select = np.asarray(np.random.randint(0, self.p, self.t_count))
        for i in t_select:
            selections.update({i : self.pop_score[i]})
        for key in sorted(selections, key = selections.get, reverse = True)[:5]:
            self.ts.append(self.pop_t[key])
    
    
    # define evolution functions
    def mutate(self):
        element = np.random.randint(0,17)
        self.m = self.pop_t[self.ts[0]].copy()
        #remove next line - only used for testing
        #print(self.m)
        self.m[element] = np.random.randint(1,27,1)
        # check for match
        m_score = 0
        for i in range(0,self.l):
            if self.m[i] == self.target[i]:
                m_score = m_score + 1
        if m_score == self.l:
            print("converged on matching phrase after " + str(self.iteration) + "iterations")
            exit
        self.pop_t1_score.update({self.q : m_score})
        if self.q < self.p:
            #remove next line - only used for testing
            #self.pop_t1.append(self.ts[0])
            self.pop_t1.update({self.q : self.m})
            #Remove next line after testing:
            #print(self.m)
            self.q = self.q + 1
    
    def crossover(self):
        element = np.random.randint(0,17)
        self.m0 = self.pop_t[self.ts[0]].copy()
        self.m1 = self.pop_t[self.ts[1]].copy()
        self.m2 = np.concatenate((self.m0[0:element], self.m1[element:]))
        self.m3 = np.concatenate((self.m1[0:element], self.m0[element:]))
        #Remove next four lines after testing:
        print(self.m0)
        print(self.m1)
        print(self.m2)
        print(self.m3)
                                
        # check for match
        m_score = 0
        for i in range(0,self.l):
            if self.m2[i] == self.target[i]:
                m_score = m_score + 1
        if m_score == self.l:
            print("converged on matching phrase after " + str(self.iteration) + "iterations")
            exit 
        if self.q < self.p:
            self.pop_t1.update({self.q : self.m2})
            self.pop_t1_score.update({self.q : m_score})
            self.q = self.q + 1  
        m_score = 0
        for i in range(0,self.l):
            if self.m3[i] == self.target[i]:
                m_score = m_score + 1
        if m_score == self.l:
            print("converged on matching phrase after " + str(self.iteration) + "iterations")
            exit
        if self.q < self.p:
            self.pop_t1.update({self.q : self.m3})
            self.pop_t1_score.update({self.q : m_score})
            self.q = self.q + 1
        
    # check size of pop_t1
    def check_pop_size(self):
        if self.q >= self.p:
            self.testing.update({self.iteration : self.pop_t.copy()})
            self.pop_t = self.pop_t1.copy()
            self.pop_t1 = {}
            self.pop_t_score = self.pop_t1_score.copy()
            self.pop_t1_score = {}
            self.iteration = self.iteration + 1
            print("Iteration " + str(self.iteration -1) + "complete : " + str(self.i_score))
            self.q = 0
     
            
    # chose crossover or mutation
    def evolution(self):
        if np.random.uniform(0,1) > self.cmratio:
        # mutate
            self.mutate()
        else:
        #crossover
            self.crossover()
            
       
    # create loop for each iteration
    def loop(self):
        self.fitness_function()
        self.tournament()
        self.evolution()
        self.check_pop_size()
    
    # define Run function
    def run(self):
        self.first_pop()
        while True:
            self.loop()
            if self.iteration > self.tolerance:
                if self.i_score[self.iteration - 1] <= self.i_score[self.iteration - 1 - self.tolerance]:
                    break
                    print("Failed to converge on matching phrase after " + self.iteration -1 + "iterations")
            
 

e =  evolutionary()
e.run()
    
        
'''    # create loop for each iteration
    def loop(self):
        self.fitness_function()
        self.tournament()
        self.evolution()
        self.check_pop_size()
        
'''  
'''        
    # Evaluate fitness of each array in pop & generate total score
    def fitness_function(self):
        self.pop = self.pop_t.copy()
        self.pop_score = {}
        for x in range(0,self.p):
            score = 0
            ar = np.asarray(self.pop[x])
            for i in range(0,self.l):
                if ar[i] == self.target[i]:
                    score = score + 1
            self.pop_score.update({x : score})
            if self.pop_score[x] == self.l:
                print("converged on matching phrase after " + str(self.iteration) + "iterations")
                exit 
        self.total_score = sum(self.pop_score)
        self.i_score.update({self.iteration : self.total_score})
        
        
        
        
                self.pop = self.pop_t.copy()
        self.pop_score = {}
        for x in range(0,self.p):
            score = np.count_nonzero(self.pop_t[x].tolist() == np.asarray(self.target))
            self.pop_score.update({x : score})
            if self.pop_score[x] == self.l:
                print("converged on matching phrase after " + str(self.iteration) + "iterations")
                exit 
        self.total_score = sum(self.pop_score)
        self.i_score.update({self.iteration : self.total_score})
'''    

