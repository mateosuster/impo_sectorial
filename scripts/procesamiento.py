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
    # impo.apply()
    # ncm = impo.iloc[0]["HS6"]
    # letra_2= impo.iloc[0]["letra1"]
    # letra_3= impo.iloc[0]["letra1"]
    cont.loc[10121,"letra1_pond"] = x
    return cont.loc[10121,"letra1_pond"]


tabla_contingencia.loc[10121, "A"] 
tabla_contingencia.loc[10121] 



