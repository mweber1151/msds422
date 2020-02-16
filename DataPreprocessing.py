# -*- coding: utf-8 -*-
"""
#######################################################
@Filename: DataExploration.py
@Description: Explore the Ascentia data 
@Project: Ascentia
@Project_Lead: Mike Weber
@Author: Joseph Engler
@Date Created: Thursday, Aug 8, 2019

Revisions:
@   August 30, 2019 by Mike Weber - add multithreading 
    and multiple flight data files   
    
    
    


########################################################"""

'''
#######################################
Import Required Libararies
#######################################
'''
import pandas as pd
import numpy as np
import os
import time
from multiprocessing.pool import ThreadPool
import pickle
import psutil
import sys
import datetime

class Preprocess():
    
    def __init__(self, filelist, colNamesFile, allNamesFile, uniqueValues):
        temp = pd.read_csv(colNamesFile, header=None)
        self.uniqueColNames = {}
        for i in range(len(temp)):
            if temp[0][i] not in self.uniqueColNames:
                self.uniqueColNames.update({temp[0][i]:temp[1][i]})
        temp = pd.read_csv(allNamesFile, header=None)
        self.allNames = list(temp[0])
        temp = pd.read_csv(uniqueValues, header=None)
        self.uniqueValues = {}
        for i in range(len(temp)):
            if temp[0][i] not in self.uniqueValues:
                self.uniqueValues.update({temp[0][i]:[]})
            
            self.uniqueValues[temp[0][i]].append(temp[1][i])  
        if sys.platform == 'linux' or sys.platform == 'linux2':
            DPLF = pd.read_csv('/rfs/public/data/Ascentia/ANA_DPLF.csv', low_memory = False)
        elif sys.platform == 'win32':
            DPLF = pd.read_csv('Z:/data/Ascentia/ANA_DPLF.csv', low_memory = False)
        Tail = 'JA804A'
        TAIL_DPLF = DPLF[DPLF['Tail'] == Tail]
        '''
        ###   Use this section if looking for messages
        TAIL_DPLF = TAIL_DPLF[~TAIL_DPLF['MessageTimeAlerted'].str.contains('Null')]
        TAIL_DPLF['MessageTimeAlerted'] = pd.to_datetime(TAIL_DPLF['MessageTimeAlerted'])
        TAIL_DPLF = TAIL_DPLF.rename(columns = {'MessageTimeAlerted': 'Timestamp', 'MessageCode': 'Code'})
        ###
        '''
        ###   Use this section if looking for Events
        TAIL_DPLF = TAIL_DPLF[~TAIL_DPLF['EventMessageTimeAlerted'].str.contains('Null')]
        TAIL_DPLF['EventMessageTimeAlerted'] = pd.to_datetime(TAIL_DPLF['EventMessageTimeAlerted'])
        TAIL_DPLF = TAIL_DPLF.rename(columns = {'EventMessageTimeAlerted': 'Timestamp', 'EventMessageCode': 'Code'})
        ###
        TAIL_DPLF = TAIL_DPLF[['Timestamp', 'Code']].reset_index(drop = True)
        self.tailDPLF = TAIL_DPLF
        self.fileList = filelist
        
    
    def ImportAndAlignData(self, filename):
        data = pd.read_csv(filename, low_memory = False)
        colsToRename = {}
        for c in data.columns:
            if c == 'Timestamp':
                continue
            if c in self.uniqueColNames:
                if self.uniqueColNames[c] == c:
                    continue
                colsToRename.update({c:self.uniqueColNames[c]})#            
        for c in colsToRename:
            data = data.rename(columns={c:colsToRename[c]})
        for c in self.allNames:
            if c not in data.columns:
                data[c] = np.zeros(len(data))
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data = pd.merge(data, messageData, how = 'left', on = 'Timestamp') 
        data = data.drop(columns=['Timestamp'], axis=1)
        data = data.loc[:,~data.columns.duplicated()]
        return data
    
    def NormalizeData(self, data):
        if sys.platform == 'linux' or sys.platform == 'linux2':
            mins = pd.read_csv('/rfs/public/Code/Ascentia/columnMinValues.csv', header=None)
            maxs = pd.read_csv('/rfs/public/Code/Ascentia/columnMaxValues.csv', header=None)
        elif sys.platform == 'win32':
            mins = pd.read_csv('Z:/Code/Ascentia/columnMinValues.csv', header=None)
            maxs = pd.read_csv('Z:/Code/Ascentia/columnMaxValues.csv', header=None)
        dd = pd.DataFrame()
        for c in data.columns:
            if c == 'Timestamp':
                continue
            else:
                try:
                    if np.issubdtype(data[c].dtype, np.number):
                        try:
                            maxV = np.max(data[c])
                            dd[c] = data[c].apply(lambda x:x/maxV)
                        except Exception as ee:
                            print(ee)
                    elif data[c].dtype.name == 'bool':
                        try:
                            dd[c] = data[c].apply(lambda x:x*1)
                        except Exception as ee:
                            print(ee)
                    else:
                        dd[c] = data[c]
                except Exception as e:
                    print(e)
        data = dd
        print('starting one-hot-encoding')
        data = self.OHEData(data)
        return data
    
    def OHEData(self, data):
        ohe_cols = {}
        colsToDrop = []
        for c in self.uniqueValues:
            colName = c
            if c in self.uniqueColNames:
                colName = self.uniqueColNames[c]
            c = colName           
            colsToDrop.append(c)
            for cl in self.uniqueValues[c]:
                ohe_cols.update({c+'_'+cl:np.zeros(len(data))})
            for i in range(len(data)):
                try:
                    val = str(data[c][i])
                    for cc in ohe_cols:
                        if c in cc and val in cc:
                            ohe_cols[cc][i] = 1
                except Exception as e:
                    print('Error - ' + str(data[c][i]))
        data = data.drop(columns=colsToDrop, axis=1)
        for c in ohe_cols:
            data[c] = ohe_cols[c]
        data = data.sort_index(axis=1)
        return data
    
    def RemoveLeadUp(self):
        t_stamps = self.tailDPLF['Timestamp'].values.astype('datetime64[ns]')
        leadUp = []

        for i in t_stamps:
            for x in range(1,100):
                ts = i - np.timedelta64(x,'s')
                leadUp.append({'Timestamp': ts, 'Code': 9999999})
        messageData = self.tailDPLF.append(leadUp)
        return messageData

        
        
    def OHEMessageData(self):
        t_stamps = self.tailDPLF['Timestamp'].values.astype('datetime64[ns]')
        leadUp = []

        for i in t_stamps:
            for x in range(1,100):
                ts = i - np.timedelta64(x,'s')
                leadUp.append({'Timestamp': ts, 'Code': 9999999})
        
        self.tailDPLF = self.tailDPLF.append(leadUp)
        
        ohe_cols = {}
        timestamps = []
    
        for t in self.tailDPLF['Timestamp'].unique():
            timestamps.append(t)
        messageData = pd.DataFrame({'Timestamp' : timestamps, 'Code' : np.zeros})
        for c in self.tailDPLF['Code'].unique():
            ohe_cols.update({'Code_' + str(c) :np.zeros(len(messageData))})
            messageData['Code_' + str(c)] = ohe_cols['Code_' + str(c)]
        for t in range(len(self.tailDPLF['Timestamp'])):
            code = self.tailDPLF['Code'][t]
            messageData['Code'][t] = 1
            messageData.loc[messageData.Timestamp == self.tailDPLF['Timestamp'][t], 'Code_' + str(code)] = 1
        return messageData
    
    def run(self):
        start_time = time.time()
        print(start_time)
        worker = ThreadPool(processes=len(self.fileList))
        threadData = worker.map(proc.ImportAndAlignData, self.fileList)
        worker.close()
        while len(threadData) < len(self.fileList):
            time.sleep(1)
        end_time = time.time()
        elapsed = (end_time - start_time)
        print(elapsed)
        data = pd.concat(threadData).reset_index()
        return data
                        
log = {}
tail = 'JA804A'
fileNo = 0
files = np.asarray(sorted(os.listdir("/usr/lfs/v0/Ascentia/Code/FileNames")))
for f in files:
    fileData = pickle.load(open('/usr/lfs/v0/Ascentia/Code/FileNames/' + f, 'rb'))
    starting_time = time.time()
    if sys.platform == 'linux' or sys.platform == 'linux2':
        proc = Preprocess(fileData, '/rfs/public/Code/Ascentia/nameChanges.csv', 
                          '/rfs/public/Code/Ascentia/uniqueColNames.csv', 
                          '/rfs/public/Code/Ascentia/columnUniqueValues.csv')
    elif sys.platform == 'win32':
        proc = Preprocess(fileData, 'Z:/Code/Ascentia/nameChanges.csv', 
                          'Z:/Code/Ascentia/uniqueColNames.csv', 
                          'Z:/Code/Ascentia/columnUniqueValues.csv')
    print('starting message data')
    messageData = proc.RemoveLeadUp()
    print('starting import & align')
    data = proc.run()
    print('starting Normalization')
    data = proc.NormalizeData(data)
    print('starting fill na')
    data = data.fillna(0)
    data = data[data['Code'] == 0]
    data.to_csv('/usr/lfs/v0/Ascentia/Code/' + f + '.csv')
    #pickle.dump(data, open('JA804A_A.pkl', 'wb'))
    print(psutil.virtual_memory())
    del data
    #print(psutil.virtual_memory())
    t_time = (time.time() - starting_time)
    fileNo += 1
    print('File number :' + str(fileNo) + ' (' + str(f) + ') ' +' Completed in: ' + str(t_time) + ' seconds')


'''
A = pickle.load(open('A.pkl', 'rb'))
starting_time = time.time()
proc = Preprocess(A, '/rfs/public/Code/Ascentia/nameChanges.csv', '/rfs/public/Code/Ascentia/uniqueColNames.csv', '/rfs/public/Code/Ascentia/columnUniqueValues.csv')
print('starting message data')
messageData = proc.RemoveLeadUp()
print('starting import & align')
data = proc.run()
print('starting Normalization')
data = proc.NormalizeData(data)
print('starting fill na')
data = data.fillna(0)
data = data[data['Code'] == 0]
data.to_csv('JA804A_A.csv')
del data
t_time = (time.time() - starting_time)
print(t_time)


B = pickle.load(open('B.pkl', 'rb'))
starting_time = time.time()
proc = Preprocess(B, '/rfs/public/Code/Ascentia/nameChanges.csv', '/rfs/public/Code/Ascentia/uniqueColNames.csv', '/rfs/public/Code/Ascentia/columnUniqueValues.csv')
print('starting message data')
messageData = proc.RemoveLeadUp()
print('starting import & align')
data = proc.run()
print('starting Normalization')
data = proc.NormalizeData(data)
print('starting fill na')
data = data.fillna(0)
data = data[data['Code'] == 0]
data.to_csv('JA804A_B.csv')
del data
t_time = (time.time() - starting_time)
print(t_time)


C = pickle.load(open('C.pkl', 'rb'))
starting_time = time.time()
proc = Preprocess(C, '/rfs/public/Code/Ascentia/nameChanges.csv', '/rfs/public/Code/Ascentia/uniqueColNames.csv', '/rfs/public/Code/Ascentia/columnUniqueValues.csv')
print('starting message data')
messageData = proc.RemoveLeadUp()
print('starting import & align')
data = proc.run()
print('starting Normalization')
data = proc.NormalizeData(data)
print('starting fill na')
data = data.fillna(0)
data = data[data['Code'] == 0]
data.to_csv('JA804A_C.csv')
t_time = (time.time() - starting_time)
print(t_time)


D = pickle.load(open('D.pkl', 'rb'))
starting_time = time.time()
proc = Preprocess(D, '/rfs/public/Code/Ascentia/nameChanges.csv', '/rfs/public/Code/Ascentia/uniqueColNames.csv', '/rfs/public/Code/Ascentia/columnUniqueValues.csv')
print('starting message data')
messageData = proc.RemoveLeadUp()
print('starting import & align')
data = proc.run()
print('starting Normalization')
data = proc.NormalizeData(data)
print('starting fill na')
data = data.fillna(0)
data = data[data['Code'] == 0]
data.to_csv('JA804A_D.csv')
t_time = (time.time() - starting_time)
print(t_time)


E = pickle.load(open('E.pkl', 'rb'))
starting_time = time.time()
proc = Preprocess(E, '/rfs/public/Code/Ascentia/nameChanges.csv', '/rfs/public/Code/Ascentia/uniqueColNames.csv', '/rfs/public/Code/Ascentia/columnUniqueValues.csv')
print('starting message data')
messageData = proc.RemoveLeadUp()
print('starting import & align')
data = proc.run()
print('starting Normalization')
data = proc.NormalizeData(data)
print('starting fill na')
data = data.fillna(0)
data = data[data['Code'] == 0]
data.to_csv('JA804A_E.csv')
t_time = (time.time() - starting_time)
print(t_time)  

F = pickle.load(open('F.pkl', 'rb'))
starting_time = time.time()
proc = Preprocess(F, '/rfs/public/Code/Ascentia/nameChanges.csv', '/rfs/public/Code/Ascentia/uniqueColNames.csv', '/rfs/public/Code/Ascentia/columnUniqueValues.csv')
print('starting message data')
messageData = proc.RemoveLeadUp()
print('starting import & align')
data = proc.run()
print('starting Normalization')
data = proc.NormalizeData(data)
print('starting fill na')
data = data.fillna(0)
data = data[data['Code'] == 0]
data.to_csv('JA804A_F.csv')
t_time = (time.time() - starting_time)
print(t_time) 

G = pickle.load(open('G.pkl', 'rb'))
starting_time = time.time()
proc = Preprocess(G, '/rfs/public/Code/Ascentia/nameChanges.csv', '/rfs/public/Code/Ascentia/uniqueColNames.csv', '/rfs/public/Code/Ascentia/columnUniqueValues.csv')
print('starting message data')
messageData = proc.RemoveLeadUp()
print('starting import & align')
data = proc.run()
print('starting Normalization')
data = proc.NormalizeData(data)
print('starting fill na')
data = data.fillna(0)
data = data[data['Code'] == 0]
data.to_csv('JA804A_G.csv')
t_time = (time.time() - starting_time)
print(t_time)     

'''