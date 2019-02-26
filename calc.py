# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 03:29:58 2019

@author: rober
"""
import os
os.chdir('C:/Users/rober/Desktop/act-remote/proyecto-sim')
import pandas as pd
from simFunc import SimUL



#data = [[25,10,0,0],[26,0,0,0],['27+',0,0,0]]        
#initPG1_H = pd.DataFrame(data,columns=['Edad',0,1,'2+'], dtype=int)
#initPG1_H = initPG1_H.set_index('Edad')
#
#data2 = [[25,10,0,0],[26,0,0,0],['27+',0,0,0]]        
#initPG1_M = pd.DataFrame(data2, columns=['Edad',0,1,'2+'], dtype=int)
#initPG1_M = initPG1_M.set_index('Edad')
#
#data3 = [[25,1],[26,1],['27+',0]]
#initPG2_H = pd.DataFrame(data3, columns=['Edad',0], dtype=int)
#initPG2_H = initPG2_H.set_index('Edad') 
#
#initPG2_M = initPG2_H.copy()
#
#data4 = [[25,1,1,1],[26,1,1,1],['27+',1,1,0]]
#decTable_H = pd.DataFrame(data4,columns=['Edad',0,1,'2+'], dtype=float)
#decTable_H = decTable_H.set_index('Edad')
#
#decTable_M = decTable_H.copy()



workbook = pd.read_excel('Tablas.xlsx', sheet_name=None) 
 
initPG1_H = workbook['inicial_H'] 
initPG1_H = initPG1_H.set_index('Edad\TS')
initPG1_M = workbook['inicial_M'] 
initPG1_M = initPG1_M.set_index('Edad\TS')

initPG2_H = workbook['ing_H'] 
initPG2_H = initPG2_H.set_index('Edad\TS')
initPG2_M = workbook['ing_M'] 
initPG2_M = initPG2_M.set_index('Edad\TS')

decTable_H = workbook['dec_H'] 
decTable_H = decTable_H.set_index('Edad\TS')
decTable_M = workbook['dec_M'] 
decTable_M = decTable_M.set_index('Edad\TS')


nsim = SimUL(decTable_H, decTable_M, initPG1_H, initPG2_H, initPG1_M, initPG2_M)

pg1T, pg2T, iact, pg1h, pg1m, pg2h, pg2m = nsim.simulate(1,30,1)

nsim.grid_plot(1, 2)
nsim.var_plot(1, 0, 0.9)







