# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:49:32 2019

@author: rober
"""
import os
os.chdir('C:/Users/rober/Desktop/act-remote/proyecto-sim')
from hmap import heatmap, annotate_heatmap, txt_remove
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class State:     
    
    def __init__(self, Id_i = None, Id_j = None, next_i = None, next_j = None, g = None, name = None, nPop = 0, 
                          transMat = None, cost = None, prem = None):
        self.Id_i = Id_i
        self.Id_j = Id_j
        self.next_i = next_i
        self.next_j = next_j
        self.g = g                          #activo o inactivo
        self.name = name
        self.nPop = nPop      
        self.transMat = transMat
        self.cost = cost
        self.prem = prem
        self.totCost = 0
        self.totPrem = 0 
          
    def compute(self):
        self.totCost = self.cost * self.nPop 
        self.totPrem = self.prem * self.prem
    
    def send(self, transTable):
        if self.Id_i == 'NA':
            transTable['NA'] += self.nPop            
        else:
            prob = np.fromiter(self.transMat.values(), dtype=float)
            popSim = np.squeeze(np.random.multinomial(self.nPop, prob, size=1))
        
            for val in zip(self.transMat.keys(), popSim):
                transTable[val[0]] += val[1]       
    
    def receive(self, transTable):
        if self.g == 1:
            self.nPop = transTable[(self.Id_i, self.Id_j, 1)]
            
        elif self.Id_i == 'NA':       
            self.nPop = transTable['NA']
        
 
    
      
class SimUL: 
     
    
    def __init__(self, decTable, initPG1, initPG2):
        self.decTable = decTable
        self.initPG1 = initPG1
        self.initPG2 = initPG2
        self.PopRes = []
        self.IngRes = []
        self.InactRes = []
        self.stateDict = {}
        self.transTable = {}
        
        
    def createStateDict(self):
        #define n*m + 1 states for g1 , intial population and (store them in a dictionary)
        n = self.initPG1.shape[0]
        m = self.initPG1.shape[1]
        
        for i in range(n):
            for j in range(m):
                
                if i == n-1:
                    if j == m-1:
                      next_i = i
                      next_j = j
                    else:
                        next_i = i
                        next_j = j + 1           
                else:
                    if j == m-1:
                        next_i = i + 1
                        next_j = j                    
                    else:
                        next_i = i + 1
                        next_j = j + 1                
            
                self.stateDict[(i, j, 1)] = \
                    State(Id_i = i, Id_j = j, next_i = next_i, next_j = next_j, \
                          g = 1, name=str(self.initPG1.index[i]) + '_' + str(self.initPG1.columns[j]))
                          
                    
        for i in range(self.initPG2.shape[0]):
            self.stateDict[(i, 0, 2)] = State(Id_i = i, g = 2, \
                               name=str(self.initPG2.index[i])) 
            
        self.stateDict['NA'] = State(Id_i = 'NA', name='Not active')  
        
        self.transTable = {el:0 for el in self.stateDict.keys()}  #assign keys for transition table
    
    
    def asgIniPop(self):
         for key in self.stateDict.keys():
            o = self.stateDict[key]
            if o.g == 1:
                o.nPop = self.initPG1.iloc[o.Id_i,o.Id_j]
            elif o.g == 2:
                o.nPop = self.initPG2.iloc[o.Id_i]
            else:
                o.nPop = 0 
    
    def zeroTransTab(self):
        self.transTable = {el:0 for el in self.stateDict.keys()}    
    
    def asgnTrMat(self):
        #transMat is a dictionary with future states and probabilities        
        for key in self.stateDict.keys():
            o = self.stateDict[key]
            if o.g == 1:
                p1 = self.decTable.iloc[o.Id_i, o.Id_j]  #trans probability
                p2 = 1 - p1                              #stay probability
                o.transMat = {(o.next_i, o.next_j, 1) : p1, "NA" : p2}
            elif o.g == 2:
                p1 = 1   #transition is deterministic 
                p2 = 0
                o.transMat = {(o.Id_i, 0, 1) : p1, "NA" : p2}  #they go to actividad = 0
            else:
                o.transMat = {"NA" : 1}
                
    
    #def asgnOpm(self):
    
    #def compute(self):  
    
    def send(self):
        for key in self.stateDict.keys():
            o = self.stateDict[key]
            o.send(self.transTable)
    
    def receive(self):
        for key in self.stateDict.keys():
            o = self.stateDict[key]
            o.receive(self.transTable)         

    def constPopMat(self):
        #we are going to build the pop matrix at a given time:
        n = self.initPG1.shape[0]
        m = self.initPG1.shape[1]
        
        mat1 = np.zeros([n,m]) 
        mat2 = np.zeros([n,1])
        inact = 0
        
        for key in self.stateDict.keys():
            o = self.stateDict[key]
            if o.g == 1:
                mat1[o.Id_i][o.Id_j] = o.nPop
            elif o.g == 2:
                mat2[o.Id_i][0] = o.nPop
            else:
                inact += o.nPop        
        
        PG1 = pd.DataFrame(mat1, dtype=int)
        PG2 = pd.DataFrame(mat2, dtype=int)
        PG1.columns = self.initPG1.columns
        PG2.columns = self.initPG2.columns
        PG1.index = self.initPG1.index
        PG2.index = self.initPG2.index
        
        return (PG1, PG2, inact)       
     
         
    def simulate(self, numSim, numStages, seed):                    
        
        np.random.seed(seed)
        self.createStateDict()
        self.asgnTrMat()     
        
        for i in range(numSim):
            
            self.asgIniPop()
            
            currPop = [self.initPG1]
            currIng = [self.initPG2]
            currInact = [self.stateDict['NA'].nPop]      
            
            for j in range(numStages):                
                               
                                
                #self.compute()  #we compute first, then transition
                
                self.send()                                   
                self.receive()
                self.zeroTransTab()  #all transitioned we need to zero the transitionTable
                
                a, b, c = self.constPopMat()                
                
                currPop.append(a)
                currIng.append(b)
                currInact.append(c)
            
            self.PopRes.append(currPop)
            self.IngRes.append(currIng) 
            self.InactRes.append(currInact)        
                
        return(self.PopRes, self.IngRes, self.InactRes)    #PopRes dimension nsimul * nstages + 1 * n * m        
    
    def grid_plot(self, simNumb, speed=2):
        fig = plt.figure( 1 )
        ax = fig.add_subplot( 111 )

        im, cbar = heatmap(self.PopRes[simNumb - 1][0].values, self.PopRes[simNumb - 1][0].index,  \
                           self.PopRes[simNumb - 1][0].columns, ax=ax, title="Pop. at initial stage", cmap="RdYlGn_r")
        texts = annotate_heatmap(im, valfmt="{x}")

        fig.show()
        im.axes.figure.canvas.draw()
        plt.pause(1/speed)

        for i in range(len(self.PopRes[simNumb - 1])-1):
            ax.set_title( 'Pop. at stage number: ' + str( i + 1) )
            txt_remove(texts)
            im.set_data( self.PopRes[simNumb - 1][i+1].values )
            texts = annotate_heatmap(im, valfmt="{x}")
            im.axes.figure.canvas.draw()
            plt.pause(1/speed)
            
        plt.close()
        
      
   
#data = [[25,3,2,8],[26,15,14,13],['27+',21,19,18]]
data = [[25,10,0,0],[26,0,0,0],['27+',0,0,0]]        
initPG1 = pd.DataFrame(data,columns=['Edad',0,1,'2+'], dtype=int)
initPG1 = initPG1.set_index('Edad')

data2 = [[25,0],[26,0],['27+',0]]
initPG2 = pd.DataFrame(data2,columns=['Edad',0], dtype=int)
initPG2 = initPG2.set_index('Edad')      
        
#data3 = [[25,0.3,0.2,0.08],[26,0.15,0.14,0.13],['27+',0.21,0.19,0.18]]
data3 = [[25,1,1,1],[26,1,1,1],['27+',1,1,0]]
decTable = pd.DataFrame(data3,columns=['Edad',0,1,'2+'], dtype=float)
decTable = decTable.set_index('Edad')

nsim = SimUL(decTable, initPG1, initPG2)
a,b,c = nsim.simulate(2,4,1)

nsim.grid_plot(1, 0.5)

























  