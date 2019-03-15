# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:49:32 2019

@author: rober
"""
from plotfun import heatmap, a_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class State:     
    
    def __init__(self, Id_i = None, Id_j = None, next_i = None, next_j = None, g = None, gen = None, \
                     name = None, nPop = 0, transMat = None):
        self.Id_i = Id_i
        self.Id_j = Id_j
        self.next_i = next_i
        self.next_j = next_j
        self.g = g          #activo o inactivo
        self.gen = gen                        
        self.name = name
        self.nPop = nPop      
        self.transMat = transMat 
    
    def send(self, transTable):
        if self.Id_i == 'NA':
            transTable['NA'] += self.nPop            
        else:
            prob = np.fromiter(self.transMat.values(), dtype=float)
            popSim = np.squeeze(np.random.multinomial(self.nPop, prob, size=1))
        
            for val in zip(self.transMat.keys(), popSim):
                transTable[val[0]] += val[1]       
    
    def receive(self, transTable):
        #if g == 2 we do not receive population
        if self.g == 1:
            if self.gen == 'H':
                self.nPop = transTable[(self.Id_i, self.Id_j, 1, 'H')]
            else:
                self.nPop = transTable[(self.Id_i, self.Id_j, 1, 'M')]            
        elif self.Id_i == 'NA':       
            self.nPop = transTable['NA']
        
 
    
      
class SimUL: 
     
    
    def __init__(self, decTable_H, decTable_M, initPG1_H, initPG2_H, initPG1_M, initPG2_M):
        self.decTable_H = decTable_H
        self.decTable_M = decTable_M
        self.initPG1_H = initPG1_H
        self.initPG2_H = initPG2_H
        self.initPG1_M = initPG1_M
        self.initPG2_M = initPG2_M        
        self.stateDict = {}
        self.transTable = {}
        
        
    def createStateDict(self):
        #define n*m + 1 states for g1 , intial population and (store them in a dictionary transTable)
        n = self.initPG1_H.shape[0]
        m = self.initPG1_H.shape[1]
        
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
            
                self.stateDict[(i, j, 1, 'H')] = \
                    State(Id_i = i, Id_j = j, next_i = next_i, next_j = next_j, \
                          g = 1, gen = 'H', \
                              name='H' + '1' + '_' + str(self.initPG1_H.index[i]) + '_' + str(self.initPG1_H.columns[j]))
                    
                self.stateDict[(i, j, 1, 'M')] = \
                    State(Id_i = i, Id_j = j, next_i = next_i, next_j = next_j, \
                          g = 1, gen = 'M', \
                              name='M' + '1' + '_' + str(self.initPG1_M.index[i]) + '_' + str(self.initPG1_M.columns[j]))                          
                    
        for i in range(self.initPG2_H.shape[0]):
            self.stateDict[(i, 0, 2, 'H')] = State(Id_i = i, g = 2, gen = 'H',\
                               name='H' + '2' + '_' + str(self.initPG2_H.index[i]))
            
            self.stateDict[(i, 0, 2, 'M')] = State(Id_i = i, g = 2, gen = 'M',\
                               name='M' + '2' + '_' + str(self.initPG2_M.index[i]))         
            
            
        self.stateDict['NA'] = State(Id_i = 'NA', name='Not active')  
        
        self.transTable = {el:0 for el in self.stateDict.keys()}  
    
    
    def asgIniPop(self):
         for key in self.stateDict.keys():
            o = self.stateDict[key]
            if o.g == 1:
                if o.gen == 'H':
                    o.nPop = int(self.initPG1_H.iloc[o.Id_i,o.Id_j])
                else:
                    o.nPop = int(self.initPG1_M.iloc[o.Id_i,o.Id_j])                
            elif o.g == 2:
                if o.gen == 'H':
                    o.nPop = int(self.initPG2_H.iloc[o.Id_i])
                else:
                    o.nPop = int(self.initPG2_M.iloc[o.Id_i])                
            else:
                o.nPop = 0 
    
    def zeroTransTab(self):
        self.transTable = {el:0 for el in self.stateDict.keys()}    
    
    def asgnTrMat(self):
        #transMat is a dictionary with future states and probabilities        
        for key in self.stateDict.keys():
            o = self.stateDict[key]
            if o.g == 1:
                if o.gen == 'H':
                    p1 = self.decTable_H.iloc[o.Id_i, o.Id_j]  #trans probability (HOMBRE)
                    p2 = 1 - p1                              #stay probability
                    o.transMat = {(o.next_i, o.next_j, 1, 'H') : p2, "NA" : p1}
                else:
                    p1 = self.decTable_M.iloc[o.Id_i, o.Id_j]  #trans probability (MUJER)
                    p2 = 1 - p1                              #stay probability
                    o.transMat = {(o.next_i, o.next_j, 1, 'M') : p2, "NA" : p1}                    
                    
            elif o.g == 2:
                p1 = 0   #transition is deterministic 
                p2 = 1
                if o.gen == 'H':
                    o.transMat = {(o.Id_i, 0, 1, 'H') : p2, "NA" : p1}  #they go to actividad = 0
                else:
                    o.transMat = {(o.Id_i, 0, 1, 'M') : p2, "NA" : p1}  #they go to actividad = 0                
            else:
                o.transMat = {"NA" : 1}
                
    
    #def asgnOpm(self):     
    
    def send(self):
        #filter states that do not have population, no population can be sent
        nklist = [k for k in self.stateDict.keys() if self.stateDict[k].nPop > 0]
        for key in nklist:
            o = self.stateDict[key]
            o.send(self.transTable)
    
    def receive(self):
        nklist = [k for k in self.stateDict.keys() if k in self.transTable.keys()]
        for key in nklist:
            o = self.stateDict[key]
            o.receive(self.transTable)         

    def constPopMat(self):
        #we are going to build the pop matrix at a given time:
        n = self.initPG1_H.shape[0]
        m = self.initPG1_H.shape[1]
        
        mat1_H = np.zeros([n,m]) 
        mat2_H = np.zeros([n,1])
        mat1_M = np.zeros([n,m]) 
        mat2_M = np.zeros([n,1])
        
        mat1_Tot = np.zeros([n,m]) 
        mat2_Tot = np.zeros([n,1])
        
        inact = 0
        nklist = [k for k in self.stateDict.keys() if self.stateDict[k].nPop > 0]  #filter, assign pop to population that only have people
        for key in nklist:
            o = self.stateDict[key]
            if o.g == 1:
                if o.gen == 'H':
                    mat1_H[o.Id_i][o.Id_j] = o.nPop
                else:
                    mat1_M[o.Id_i][o.Id_j] = o.nPop
                mat1_Tot[o.Id_i][o.Id_j] += o.nPop  #We have 2 mat refering to the same index
            elif o.g == 2:
                if o.gen == 'H':
                    mat2_H[o.Id_i][0] = o.nPop
                else:
                    mat2_M[o.Id_i][0] = o.nPop
                mat2_Tot[o.Id_i][0] += o.nPop  #We have 2 mat refering to the same index
            else:
                inact += o.nPop        
        
        PG1_H = pd.DataFrame(mat1_H, dtype=int); PG2_H = pd.DataFrame(mat2_H, dtype=int)
        PG1_M = pd.DataFrame(mat1_M, dtype=int); PG2_M = pd.DataFrame(mat2_M, dtype=int)
        PG1_Tot = pd.DataFrame(mat1_Tot, dtype=int)      
        PG2_Tot = pd.DataFrame(mat2_Tot, dtype=int)
        
        PG1_H.columns = self.initPG1_H.columns; PG1_H.index = self.initPG1_H.index
        PG1_M.columns = self.initPG1_M.columns; PG1_M.index = self.initPG1_M.index
        PG2_H.columns = self.initPG2_H.columns; PG2_H.index = self.initPG2_H.index
        PG2_M.columns = self.initPG2_M.columns; PG2_M.index = self.initPG2_M.index
        
        PG1_Tot.columns = self.initPG1_H.columns; PG1_Tot.index = self.initPG1_H.index
        PG2_Tot.columns = self.initPG2_H.columns; PG2_Tot.index = self.initPG2_H.index
        
        return (PG1_Tot, PG2_Tot, inact, PG1_H, PG1_M, PG2_H, PG2_M)    
    
    #def compute(self):  (THIS DEPENDS ON THE MATRIX CONSTRUCTION)
     
         
    def simulate(self, numSim, numStages, seed):   

        self.FPG1_T = []
        self.FPG2_T = []
        self.Finact = []
        self.FPG1_H = []
        self.FPG1_M = []
        self.FPG2_H = []
        self.FPG2_M = []                     
        
        np.random.seed(seed)
        self.createStateDict()
        self.asgnTrMat()     
        
        for i in range(numSim):
            
            print('iteration number {0} of {1}'.format(i+1,numSim))
            
            self.asgIniPop()
            
            currPG1_T = [self.initPG1_H + self.initPG1_M]
            currPG2_T = [self.initPG2_H + self.initPG2_M]
            currInact = [self.stateDict['NA'].nPop]
            currPG1_H = [self.initPG1_H] ; currPG2_H = [self.initPG2_H]
            currPG1_M = [self.initPG1_M] ; currPG2_M = [self.initPG2_M]                 
            
            for j in range(numStages):                           
                                
                #self.compute()  #we compute first, then transition
                
                self.send()                                   
                self.receive()
                self.zeroTransTab()  #all transitioned we need to zero the transitionTable
                
                pg1T, pg2T, iact, pg1h, pg1m, pg2h, pg2m = self.constPopMat()                
                
                currPG1_T.append(pg1T)
                currPG2_T.append(pg2T) 
                currInact.append(iact) 
                currPG1_H.append(pg1h)  ; currPG2_H.append(pg2h) 
                currPG1_M.append(pg1m)  ; currPG2_M.append(pg2m) 
            
            self.FPG1_T.append(currPG1_T) 
            self.FPG2_T.append(currPG2_T) 
            self.Finact.append(currInact) 
            self.FPG1_H.append(currPG1_H) ; self.FPG2_H.append(currPG2_H)
            self.FPG1_M.append(currPG1_M) ; self.FPG2_M.append(currPG2_M)                                  
                
        return(self.FPG1_T, self.FPG2_T, self.Finact, \
                   self.FPG1_H, self.FPG1_M, self.FPG2_H, self.FPG2_M)    #FPG1 dimension nsimul * nstages + 1 * n * m  
       
    
    def yieldCost(self, PremTable, CostTable, popSim):
        #popSim is the population for all stages at the nth simulation        
        for pop in popSim:
            c = (pop.values * CostTable.values).sum()
            p = (pop.values * PremTable.values).sum()
            
            yield c, p
            
    def compCosts(self, PremTableH, CostTableH, PremTableM, CostTableM, iCosto, iPrima, iInteres):
        
        CostH = []
        PremH = []
        CostM = []
        PremM = [] 
        
        CostTot = []
        PremTot = []
        TotDif = []
        
        for simH, simM in zip(self.FPG1_H, self.FPG1_M):
            
            CostHsim = 0; PremHsim = 0; CostMsim = 0; PremMsim = 0
            
            t = 0.5
            for c,p in self.yieldCost(PremTableH, CostTableH, simH):
                CostHsim += c*((1+iCosto)/(1+iInteres))**(t)
                PremHsim += p*((1+iPrima)/(1+iInteres))**(t)
                t += 1
            
            t = 0.5
            for c,p in self.yieldCost(PremTableM, CostTableM, simM):
                CostMsim += c*((1+iCosto)/(1+iInteres))**(t)
                PremMsim += p*((1+iPrima)/(1+iInteres))**(t)
                t += 1
                
            CostH.append(CostHsim)
            PremH.append(PremHsim)
            CostM.append(CostMsim)
            PremM.append(PremMsim)
            
            CostTot.append(CostHsim + CostMsim)
            PremTot.append(PremHsim + PremMsim)
            TotDif.append(PremHsim + PremMsim - CostHsim - CostMsim)
        
        return (TotDif, CostTot, PremTot, CostH, PremH, CostM, PremM)   
    
    def compCosts2(self, PremTableH, CostTableH, PremTableM, CostTableM, iCosto, iPrima, iInteres):
        
        CostH = []
        PremH = []
        CostM = []
        PremM = [] 
        
        CostTot = []
        PremTot = []
        TotDif = []
        
        for simH, simM in zip(self.FPG1_H, self.FPG1_M):
            
            CostHsim = np.array([]); PremHsim = np.array([]); CostMsim = np.array([]); PremMsim = np.array([])
            
            t = 0.5
            for c,p in self.yieldCost(PremTableH, CostTableH, simH):
                CostHsim = np.append(CostHsim, c*((1+iCosto)/(1+iInteres))**(t))
                PremHsim = np.append(PremHsim, p*((1+iPrima)/(1+iInteres))**(t))
                t += 1
            
            t = 0.5
            for c,p in self.yieldCost(PremTableM, CostTableM, simM):
                CostMsim = np.append(CostMsim, c*((1+iCosto)/(1+iInteres))**(t))
                PremMsim = np.append(PremMsim, p*((1+iPrima)/(1+iInteres))**(t))
                t += 1
                
            CostH.append(CostHsim)
            PremH.append(PremHsim)
            CostM.append(CostMsim)
            PremM.append(PremMsim)
            
            CostTot.append(CostHsim + CostMsim)
            PremTot.append(PremHsim + PremMsim)
            TotDif.append(PremHsim + PremMsim - CostHsim - CostMsim)
        
        return (TotDif, CostTot, PremTot, CostH, PremH, CostM, PremM)  
    
    def grid_plot(self, simNumb, speed=2):
        data = self.FPG1_T[simNumb - 1][0].values
        col_values = self.FPG1_T[simNumb - 1][0].index
        row_values = self.FPG1_T[simNumb - 1][0].columns
        
        fig = plt.figure( 1 )
        ax = fig.add_subplot( 111 )   
        
        im = heatmap(data, col_values,  \
                           row_values, cmap="gist_yarg", ax=ax, title="Pop. at initial stage")
        
        fig.show()
        im.axes.figure.canvas.draw()
        plt.pause(1/speed)

        for i in range(len(self.FPG1_T[simNumb - 1])-1):
            ax.set_title( 'Pop. at stage number: ' + str( i + 1) )            
            im.set_data( self.FPG1_T[simNumb - 1][i+1].values )
            im.axes.figure.canvas.draw()
            plt.pause(1/speed)
            
        plt.close()
        
    def var_plot(self, simNumb, axis, speed):
        xlim = 0
        for i in range(len(self.FPG1_T[simNumb - 1])):
            t = max(max(self.FPG1_H[simNumb - 1][i].sum(axis=axis).values) , max(self.FPG1_M[simNumb - 1][i].sum(axis=axis).values))
            xlim = max(t, xlim)
        
        y1 = self.FPG1_H[simNumb - 1][0].sum(axis=axis).values
        y2 = self.FPG1_M[simNumb - 1][0].sum(axis=axis).values
        
        if axis == 0:
            name = 'ts '
            x = self.FPG1_H[simNumb - 1][0].columns
        else:
            name = 'age '
            x = self.FPG1_H[simNumb - 1][0].index
            
        title = "Pop. by " + name + "at initial stage"
        
        fig, axes = plt.subplots(ncols=2, sharey=True)
        
        fig, axes = a_plot(fig, axes, x,y1,y2,xlim,title)
        
        plt.show()
        
        plt.pause(1/speed)
        
        for i in range(len(self.FPG1_T[simNumb - 1])-1):
            axes[0].clear()
            axes[1].clear()

            y1 = self.FPG1_H[simNumb - 1][i+1].sum(axis=axis).values
            y2 = self.FPG1_M[simNumb - 1][i+1].sum(axis=axis).values
            
            title = "Pop. by " + name + "at stage number: " + str( i + 1)
            
            fig, axes = a_plot(fig, axes, x,y1,y2,xlim,title)
            
            plt.pause(1/speed)
        
        plt.close()
      
   
    def age_evol_plot(self, simNumb):
        
        simNumb = 1
        tot_age = []
        stages = list(range(len(self.FPG1_T[simNumb - 1])))

        for i in stages:
            series = self.FPG1_T[simNumb - 1][i].sum(axis = 1)
            ages = np.append(np.array(series.index[:-1], dtype=int), int(series.index[-1].replace('+', '')))
            values = np.array(series.values, dtype = int)
            mean_age = round(sum((ages * values))/sum(values), 2)
            tot_age.append(mean_age)


        plt.style.use('seaborn-whitegrid')

        fig = plt.figure()
        ax = plt.axes()

        plt.title("Evolución de la Edad Promedio")
        plt.xlabel("Etapa")
        plt.ylabel("Edad Promedio")

        ax.plot(stages, tot_age, color='blue')



























  