# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 03:29:58 2019

@author: rober
"""
import os
os.chdir('C:/Users/rober/Desktop/act-remote/proyecto-sim/programa')
import pandas as pd
from simFunc import SimUL

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

pg1T, pg2T, iact, pg1h, pg1m, pg2h, pg2m = nsim.simulate(10,30,1)


nsim.grid_plot(1, 2)
nsim.var_plot(1, 1, 0.9)

#con este comando se le da en numero de la simulacion que queremos y nos devuelve un grafico de como va cambiando
# la edad promedio en cada etapa
nsim.age_evol_plot(1)

#Para calcular los costos cargamos las tablas:
PremTableH = workbook['PremTableH'] 
PremTableH = PremTableH.set_index('Edad\TS')
CostTableH = workbook['CostTableH'] 
CostTableH = CostTableH.set_index('Edad\TS')

PremTableM = workbook['PremTableM'] 
PremTableM = PremTableM.set_index('Edad\TS')
CostTableM = workbook['CostTableM'] 
CostTableM = CostTableM.set_index('Edad\TS')

def estabilize_prem(edad, df):
    #edad puede ser un numero o un caracter ex: '100+'
    ep = df.index.get_loc(edad)
    df2 = df.copy()
    df2.iloc[ep+1:,:] = list(df2.iloc[ep,:])
    return df2
    
PremTableM2 = estabilize_prem(25, PremTableM)
#PremTableM2 es una tabla de primas con la prima estabilizada desde en este caso 25 anios

#Parametros tabla primas hombres, tabla costos hombres, "" mujeres, tasa incremento en costos, tasa inc primas, tasa interes
res = nsim.compCosts(PremTableH, CostTableH, PremTableM, CostTableM, 0.03, 0.03, 0.07)
res[0]

#compCosts2 no entrega la suma sino el costo en valor presente de cada simulacion
#Parametros tabla primas hombres, tabla costos hombres, "" mujeres, tasa incremento en costos, tasa inc primas, tasa interes
res2 = nsim.compCosts2(PremTableH, CostTableH, PremTableM, CostTableM, 0.03, 0.03, 0.07)
res2[0]












#import os
#os.chdir('C:/Users/rober/Desktop/act-remote/proyecto-sim')
#import pandas as pd
#from simFunc import SimUL
#
#data = [[25,10,0,0],[26,0,0,0],['27+',0,0,0]]        
#initPG1_H = pd.DataFrame(data,columns=['Edad',0,1,'2+'], dtype=int)
#initPG1_H = initPG1_H.set_index('Edad')
#
#data2 = [[25,0,0,0],[26,0,0,0],['27+',0,0,0]]        
#initPG1_M = pd.DataFrame(data2, columns=['Edad',0,1,'2+'], dtype=int)
#initPG1_M = initPG1_M.set_index('Edad')
#
#data3 = [[25,0],[26,0],['27+',0]]
#initPG2_H = pd.DataFrame(data3, columns=['Edad',0], dtype=int)
#initPG2_H = initPG2_H.set_index('Edad') 
#
#initPG2_M = initPG2_H.copy()
#
#data4 = [[25,0,0,0],[26,0,0,0],['27+',0,0,1]]
#decTable_H = pd.DataFrame(data4,columns=['Edad',0,1,'2+'], dtype=float)
#decTable_H = decTable_H.set_index('Edad')
#
#decTable_M = decTable_H.copy()
#
#PremTableH = initPG1_M.copy()
#CostTableH = initPG1_M.copy()
#PremTableM = initPG1_M.copy()
#CostTableM = initPG1_M.copy()
#
#CostTableH[0][0] = 10
#CostTableH[1][1] = 10
#
#nsim = SimUL(decTable_H, decTable_M, initPG1_H, initPG2_H, initPG1_M, initPG2_M)
#
#pg1T, pg2T, iact, pg1h, pg1m, pg2h, pg2m = nsim.simulate(2,4,1)
#
#res = nsim.compCosts(PremTableH, CostTableH, PremTableM, CostTableM, 0.03, 0.00, 0.10)
#res[0]