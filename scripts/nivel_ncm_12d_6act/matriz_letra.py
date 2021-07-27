# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:44:28 2021

@author: igalk
"""

from tqdm import tqdm
import pandas as pd
import numpy as np

def def_insumo_matriz(raw_data):
    
    #creamos df para guardar los insumos de la matriz
    for_fill = pd.DataFrame()
    for_fill ["cuit"]=""
    for_fill ["hs6_d12"]=""
    for_fill ["valor_pond"]=""    
    for_fill ["si"]=""    
    for_fill ["sd"]=""
    for_fill ["ue_dest"]=""

    for a in tqdm(range(len(raw_data))):
    # for a in tqdm(range(1,100)):
        for b, c, d, e in zip(["letra1", "letra2", "letra3"],
                              ["vta_bk", "vta_bk2", "vta_bk3"],
                              ["vta_sec","vta_sec2", "vta_sec3"],
                              ["letra1_pond", "letra2_pond", "letra3_pond"]):
            
            if raw_data.iloc[a][b] == "G":
                if raw_data.iloc[a][c] == 0:
                    letra_sd= raw_data.iloc[a][b]
                    
                elif raw_data.iloc[a][d]==0:
                     letra_sd = "CONS"
                     
                else:
                   letra_sd = None #np.nan #asigno por probabilidad
                       
            else:
                letra_sd = raw_data.iloc[a][b]
               
            
            values = {'cuit': raw_data.iloc[a]["CUIT_IMPOR"],
                      "hs6_d12": raw_data.iloc[a]["HS6_d12"],
                      "valor_pond":raw_data.iloc[a]["valor"]*raw_data.iloc[a][e] ,
                      "si": raw_data.iloc[a]["letra1"],
                      "sd": letra_sd,
                      "ue_dest": "nan"}
            #print (values)
            for_fill= for_fill.append(values, ignore_index=True)
            # print (a/len(raw_data),"%")
    return for_fill
    
def def_matriz_c_prob(prob):
    
    calc_prob = prob.copy()
    calc_none = prob.copy()
    sisd_final = prob.copy()
    
   
    calc_prob = calc_prob[calc_prob["sd"].notnull()]
    calc_prob = calc_prob[calc_prob["sd"]!="G"].groupby(["hs6_d12","sd"])["valor_pond"].agg("sum")
    calc_prob = calc_prob.groupby(level=0).apply(lambda x: x/float(x.sum())).reset_index().rename(columns={"valor_pond":"valor_prob"})
   
    
    calc_none = calc_none[calc_none["sd"].isnull()]
    calc_none = pd.merge(left=calc_none.drop("sd", axis=1), right=calc_prob, left_on="hs6_d12", right_on = "hs6_d12", how="left")
    calc_none["valor_pond"] = calc_none["valor_pond"] * calc_none["valor_prob"] 
    calc_none.drop("valor_prob", axis=1, inplace=True)
        
    matriz_sisd_final = pd.concat([sisd_final[sisd_final["sd"].notnull()], calc_none])
    
    return matriz_sisd_final
                   
    
def to_matriz(matriz_sisd_final):
    #probar con index = hs6
    z = pd.pivot_table(matriz_sisd_final, values='valor_pond', index=['si'], columns=['sd'], aggfunc=np.sum, fill_value=0) 
    cols=list(z.columns.values)
    cols.pop(cols.index("CONS"))
    z=z[cols+["CONS"]] #ubicacion del consumo ultima colummna
    
    z= z.append(pd.Series(name='T')) #imputacion de T
    z= z.replace(np.nan,0)
    
    z= z.append(pd.Series(name='CONS')) #imputacion de CONS
    z= z.replace(np.nan,0)
    
    return z
