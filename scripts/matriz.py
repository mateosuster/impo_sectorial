# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:44:28 2021

@author: igalk
"""

import pandas as pd

def def_insumo_matriz(for_fill, raw_data):
#    for a in range(len(raw_data)):
    for a in range(1:100):
        for b, c, d, e in zip(["letra1", "letra2", "letra3"],
                              ["vta_bk", "vta_bk2", "vta_bk3"],
                              ["vta_sec","vta_sec2", "vta_sec3"],
                              ["letra1_pond", "letra2_pond", "letra3_pond"]):
            

            if raw_data.iloc[a][b] == "G":
                if raw_data.iloc[a][c] == 0:
                    letra_sd= raw_data.iloc[a][b]
                    print (letra_sd)
                elif raw_data.iloc[a][d]==0:
                     letra_sd = "CONS"
                     print (letra_sd)
                     
                else:
                   letra_sd = None #np.nan #asigno por probabilidad
                   print (letra_sd)
                   
            else:
                letra_sd = raw_data.iloc[a][b]
                print (letra_sd)
            
            values = {'cuit': raw_data.iloc[a]["CUIT_IMPOR"],
                      "hs6": raw_data.iloc[a]["HS6"],
                      "valor_pond":raw_data.iloc[a]["valor"]*raw_data.iloc[a][e] ,
                      "si": raw_data.iloc[a]["letra1"],
                      "sd": letra_sd,
                      "ue_dest": "nan"}
            print (values)
            for_fill= for_fill.append(values, ignore_index=True)
            print (a/len(raw_data),"%")
    return for_fill
    
    # if x["importador"] == "G":
    #     if x["vta_bk"]==0:
    #         return x['importador']
    #     elif x["vta_sec"]==0:
    #         return "CONS"
    #     else:
    #         return None #np.nan #asigno por probabilidad
        
    # elif x['importador'] != "G":
    #      return x['importador']
     
        
