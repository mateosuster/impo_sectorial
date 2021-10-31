# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:44:28 2021

@author: igalk
"""
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import datatable as dt
os.chdir("C:/Users/igalk/OneDrive/Documentos/laburo/CEP/procesamiento impo/nuevo1/impo_sectorial/scripts/nivel_ncm_12d_6act")

def def_insumo_matriz(raw_data):
    raw_data = pd.read_csv("../data/resultados/impo_con_ponderaciones_12d_6act_post_ml.csv")
    raw_data
    #creamos df para guardar los insumos de la matriz
    for_fill = pd.DataFrame()
    for_fill["cuit"]=""
    for_fill["hs6_d12"]=""
    for_fill["valor_pond"]=""
    for_fill["si"]=""
    for_fill["sd"]=""
    for_fill["ue_dest"]=""
    for_fill = dt.Frame(for_fill)
    for_fill["cuit"] = int
    for_fill["hs6_d12"] = str
    for_fill["valor_pond"] = float
    for_fill["si"] = str
    for_fill["sd"] = str
    for_fill["ue_dest"] = str

    letra1 = raw_data.columns.get_loc("letra1") + 1
    letra2 = raw_data.columns.get_loc("letra2") + 1
    letra3 = raw_data.columns.get_loc("letra3") + 1
    letra4 = raw_data.columns.get_loc("letra4") + 1
    letra5 = raw_data.columns.get_loc("letra5") + 1
    letra6 = raw_data.columns.get_loc("letra6") + 1

    vta_bk = raw_data.columns.get_loc("vta_bk") + 1
    vta_bk2 = raw_data.columns.get_loc("vta_bk2") + 1
    vta_bk3 = raw_data.columns.get_loc("vta_bk3") + 1
    vta_bk4 = raw_data.columns.get_loc("vta_bk4") + 1
    vta_bk5 = raw_data.columns.get_loc("vta_bk5") + 1
    vta_bk6 = raw_data.columns.get_loc("vta_bk6") + 1

    vta_sec = raw_data.columns.get_loc("vta_sec") + 1
    vta_sec2 = raw_data.columns.get_loc("vta_sec2") + 1
    vta_sec3 = raw_data.columns.get_loc("vta_sec3") + 1
    vta_sec4 = raw_data.columns.get_loc("vta_sec4") + 1
    vta_sec5 = raw_data.columns.get_loc("vta_sec5") + 1
    vta_sec6 = raw_data.columns.get_loc("vta_sec6") + 1

    letra1_pond = raw_data.columns.get_loc("letra1_pond") + 1
    letra2_pond = raw_data.columns.get_loc("letra2_pond") + 1
    letra3_pond = raw_data.columns.get_loc("letra3_pond") + 1
    letra4_pond = raw_data.columns.get_loc("letra4_pond") + 1
    letra5_pond = raw_data.columns.get_loc("letra5_pond") + 1
    letra6_pond = raw_data.columns.get_loc("letra6_pond") + 1



    #for a in tqdm(range(len(raw_data))):
    # for a in tqdm(range(1,100)):
        #for b, c, d, e in zip(["letra1", "letra2", "letra3","letra4", "letra5", "letra6"],
                              # ["vta_bk", "vta_bk2", "vta_bk3","vta_bk4", "vta_bk5", "vta_bk6"],
                              # ["vta_sec","vta_sec2", "vta_sec3","vta_sec4","vta_sec5", "vta_sec6"],
                              # ["letra1_pond", "letra2_pond", "letra3_pond","letra4_pond", "letra5_pond", "letra6_pond"]):

    for a in tqdm(raw_data.itertuples()):
        for b, c, d, e in zip([letra1, letra2, letra3,letra4, letra5, letra6],
                              [vta_bk, vta_bk2, vta_bk3,vta_bk4, vta_bk5, vta_bk6],
                              [vta_sec, vta_sec2, vta_sec3, vta_sec4, vta_sec5, vta_sec6],
                              [letra1_pond, letra2_pond, letra3_pond, letra4_pond, letra5_pond, letra6_pond]):


            # t = a[b]
            # y = a[c]
            # u = a[d]
            # i = a[e]

            # [b], a[c], a[d], a[e])


            # if raw_data.iloc[a][b] == "G":
            #     if raw_data.iloc[a][c] == 0:
            #         letra_sd= raw_data.iloc[a][b]

            if a[b] == "G":
                if a[c] == 0:
                    letra_sd = a[b]

                # elif raw_data.iloc[a][d]==0:
                #      letra_sd = "CONS"

                elif a[d] ==0:
                    letra_sd = "CONS"

                # else:
                #    letra_sd = None #np.nan #asigno por probabilidad

                else:
                   letra_sd = None #np.nan #asigno por probabilidad


            # else:
            #     letra_sd = raw_data.iloc[a][b]

            else:
                letra_sd = a[b]


            # values = {'cuit': raw_data.iloc[a]["cuit"],
            #           "hs6_d12": raw_data.iloc[a]["HS6_d12"],
            #           "valor_pond":raw_data.iloc[a]["valor"]*raw_data.iloc[a][e] ,
            #           "si": raw_data.iloc[a]["letra1"],
            #           "sd": letra_sd,
            #           "ue_dest": "nan"}

            values = dt.Frame({'cuit': [a[1]],
                      "hs6_d12": [a[3]],
                      "valor_pond": [a[6] * a[e]],
                      "si": [a[20]],
                      "sd": [letra_sd],
                      "ue_dest": ["nan"]})

            # for_fill= for_fill.append(values, ignore_index=True)
            for_fill.rbind([values])
            #print (for_fill)
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
