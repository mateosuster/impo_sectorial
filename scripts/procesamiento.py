# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:46:35 2021

@author: igalk
"""
import pandas as pd
import numpy as np

def def_contingencia(join_impo_clae_bec_bk_comercio):
    impo_by_product_1 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra1" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra1": "letra"})
    impo_by_product_2 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra2" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra2": "letra"})
    impo_by_product_3 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra3" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra3": "letra"})
    
    impo_by_product = impo_by_product_1.append([impo_by_product_2,impo_by_product_3], ignore_index=True)
    table = pd.pivot_table(impo_by_product, values='valor', index=['HS6'], columns=['letra'], aggfunc=np.sum, fill_value=0)
    return table

def def_join_impo_clae_bec_bk_comercio_pond(ncm_act_pond, tabla_contingencia):
    x = ncm_act_pond.copy()
    x["letra1_pond"] = np.nan
    x["letra2_pond"] = np.nan
    x["letra3_pond"] = np.nan
    
    
    return x
    
def def_calc_pond(impo,cont): #quedó a medio camino
    for a in range(len(impo)):
        cuit= impo.iloc[a]["CUIT_IMPOR"]
        letra_1= impo.iloc[a]["letra1"]
        letra_2= impo.iloc[a]["letra2"]
        letra_3= impo.iloc[a]["letra3"]
        print(cuit, letra_1, letra_2, letra_3)
        
        x=[]
        for b in ([letra_1, letra_2, letra_3]):
            ncm = impo.iloc[a]["HS6"]
            ncm_val = cont.loc[ncm][b]
            x.append(ncm_val)
        
        total=x[0]+x[1]+x[2]
        act1_pond=x[0]/total
        act2_pond=x[1]/total
        act3_pond=x[2]/total
        impo.at[a, "letra1_pond"] = act1_pond
        impo.at[a, "letra2_pond"] = act2_pond
        impo.at[a, "letra3_pond"] = act3_pond
        print(ncm, x, total, act1_pond, act2_pond, act3_pond)



