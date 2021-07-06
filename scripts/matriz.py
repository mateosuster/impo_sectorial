# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:44:28 2021

@author: igalk
"""

from tqdm import tqdm
import pandas as pd

def def_insumo_matriz(for_fill, raw_data):
    for a in tqdm(range(len(raw_data))):
    # for a in tqdm(range(1,100)):
        for b, c, d, e, f in zip(["letra1", "letra2", "letra3"],
                              ["vta_bk", "vta_bk2", "vta_bk3"],
                              ["vta_sec","vta_sec2", "vta_sec3"],
                              ["actividad1_pond", "actividad2_pond", "actividad3_pond"],
                              ["actividad1", "actividad2", "actividad3"]):
            
            if raw_data.iloc[a][b] == "G":
                if raw_data.iloc[a][c] == 0:
                    letra_sd= raw_data.iloc[a][f]
                    
                elif raw_data.iloc[a][d]==0:
                     letra_sd = "CONS"
                     
                else:
                   letra_sd = None #np.nan #asigno por probabilidad
                       
            else:
                letra_sd = raw_data.iloc[a][f]
               
            
            values = {'cuit': raw_data.iloc[a]["CUIT_IMPOR"],
                      "hs6": raw_data.iloc[a]["HS6"],
                      "valor_pond":raw_data.iloc[a]["valor"]*raw_data.iloc[a][e] ,
                      "si": raw_data.iloc[a]["actividad1"],
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
    calc_prob = calc_prob[calc_prob["sd"]!=45] #excluyendo comercio
    calc_prob = calc_prob[calc_prob["sd"]!=46] #excluyendo comercio
    calc_prob = calc_prob[calc_prob["sd"]!=47] #excluyendo comercio
    calc_prob = calc_prob.groupby(["hs6","sd"])["valor_pond"].agg("sum")
    calc_prob = calc_prob.groupby(level=0).apply(lambda x: x/float(x.sum())).reset_index().rename(columns={"valor_pond":"valor_prob"})
   
    
    calc_none = calc_none[calc_none["sd"].isnull()]
    calc_none = pd.merge(left=calc_none.drop("sd", axis=1), right=calc_prob, left_on="hs6", right_on = "hs6", how="left")
    calc_none["valor_pond"] = calc_none["valor_pond"] * calc_none["valor_prob"] 
    calc_none.drop("valor_prob", axis=1, inplace=True)
        
    matriz_sisd_final = pd.concat([sisd_final[sisd_final["sd"].notnull()], calc_none])
    
    return matriz_sisd_final
                   
    
    
    # z = pd.pivot_table(prob, values='valor_pond', index=['hs6'], columns=['sd'], aggfunc=np.sum, fill_value=0)

    
    
    # if x["importador"] == "G":
    #     if x["vta_bk"]==0:
    #         return x['importador']
    #     elif x["vta_sec"]==0:
    #         return "CONS"
    #     else:
    #         return None #np.nan #asigno por probabilidad
        
    # elif x['importador'] != "G":
    #      return x['importador']
     
        
