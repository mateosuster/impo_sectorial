# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:46:35 2021

@author: igalk
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def def_contingencia(join_impo_clae_bec_bk_comercio):
    
    impo_by_product_1 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra1" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra1": "letra"})
    impo_by_product_2 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra2" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra2": "letra"})
    impo_by_product_3 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra3" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra3": "letra"})
    impo_by_product_4 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra4" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra4": "letra"})
    impo_by_product_5 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra5" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra5": "letra"})
    impo_by_product_6 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra6" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra6": "letra"})
    
    impo_by_product = impo_by_product_1.append([impo_by_product_2,impo_by_product_3,impo_by_product_4,impo_by_product_5,impo_by_product_6], ignore_index=True)
    table = pd.pivot_table(impo_by_product, values='valor', index=['HS6_d12'], columns=['letra'], aggfunc=np.sum, fill_value=0)
        
    return table

def def_join_impo_clae_bec_bk_comercio_pond(ncm_act_pond, tabla_contingencia):
    x = ncm_act_pond.copy()
    x["letra1_pond"] = np.nan
    x["letra2_pond"] = np.nan
    x["letra3_pond"] = np.nan
    x["letra4_pond"] = np.nan
    x["letra5_pond"] = np.nan
    x["letra6_pond"] = np.nan
    
    return x
    
def def_calc_pond(impo,cont):
    join_final = impo.copy()
    for a in tqdm(range(len(join_final))):
        # cuit= join_final.iloc[a]["CUIT_IMPOR"]
        letra_1= join_final.iloc[a]["letra1"]
        letra_2= join_final.iloc[a]["letra2"]
        letra_3= join_final.iloc[a]["letra3"]
        letra_4= join_final.iloc[a]["letra4"]
        letra_5= join_final.iloc[a]["letra5"]
        letra_6= join_final.iloc[a]["letra6"]
        
        # print(cuit, letra_1, letra_2, letra_3)
        
        x=[]
        for b in [letra_1, letra_2, letra_3, letra_4, letra_5, letra_6]:
            ncm = join_final.iloc[a]["HS6_d12"]
            ncm_val = cont.loc[ncm][b]
            x.append(ncm_val)
        
        total=x[0]+x[1]+x[2]+x[3]+x[4]+x[5]
        act1_pond=x[0]/total
        act2_pond=x[1]/total
        act3_pond=x[2]/total
        act4_pond=x[3]/total
        act5_pond=x[4]/total
        act6_pond=x[5]/total
        
        join_final.at[a, "letra1_pond"] = act1_pond
        join_final.at[a, "letra2_pond"] = act2_pond
        join_final.at[a, "letra3_pond"] = act3_pond
        join_final.at[a, "letra4_pond"] = act4_pond
        join_final.at[a, "letra5_pond"] = act5_pond
        join_final.at[a, "letra6_pond"] = act6_pond
        
        # print(ncm, x, total, act1_pond, act2_pond, act3_pond)
    return join_final

