# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:46:35 2021

@author: igalk
"""
import pandas as pd
import numpy as np

def def_contingencia(join_impo_clae_bec_bk_comercio):
    #a nivel letra
    # impo_by_product_1 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra1" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra1": "letra"})
    # impo_by_product_2 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra2" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra2": "letra"})
    # impo_by_product_3 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra3" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra3": "letra"})
    
    join_impo_clae_bec_bk_comercio_2d = join_impo_clae_bec_bk_comercio.copy()
    
    join_impo_clae_bec_bk_comercio_2d['actividad1'] = join_impo_clae_bec_bk_comercio_2d['actividad1'].astype(str).str[:2]
    join_impo_clae_bec_bk_comercio_2d['actividad2'] = join_impo_clae_bec_bk_comercio_2d['actividad2'].astype(str).str[:2]
    join_impo_clae_bec_bk_comercio_2d['actividad3'] = join_impo_clae_bec_bk_comercio_2d['actividad3'].astype(str).str[:2]
    #a dos digitos
    impo_by_product_1 = join_impo_clae_bec_bk_comercio_2d.groupby(["HS6", "actividad1" ], as_index = False).agg({'valor': sum}).rename(columns = {"actividad1": "actividad"})
    impo_by_product_2 = join_impo_clae_bec_bk_comercio_2d.groupby(["HS6", "actividad2" ], as_index = False).agg({'valor': sum}).rename(columns = {"actividad2": "actividad"})
    impo_by_product_3 = join_impo_clae_bec_bk_comercio_2d.groupby(["HS6", "actividad3" ], as_index = False).agg({'valor': sum}).rename(columns = {"actividad3": "actividad"})
    
    impo_by_product = impo_by_product_1.append([impo_by_product_2,impo_by_product_3], ignore_index=True)
    table = pd.pivot_table(impo_by_product, values='valor', index=['HS6'], columns=['actividad'], aggfunc=np.sum, fill_value=0)
    return table

def def_join_impo_clae_bec_bk_comercio_pond(ncm_act_pond, tabla_contingencia):
    x = ncm_act_pond.copy()
    x["letra1_pond"] = np.nan
    x["letra2_pond"] = np.nan
    x["letra3_pond"] = np.nan
    
    
    return x
    
def def_calc_pond(impo,cont):
    join_final = impo.copy()
    for a in range(len(join_final)):
        cuit= join_final.iloc[a]["CUIT_IMPOR"]
        letra_1= join_final.iloc[a]["letra1"]
        letra_2= join_final.iloc[a]["letra2"]
        letra_3= join_final.iloc[a]["letra3"]
        print(cuit, letra_1, letra_2, letra_3)
        
        x=[]
        for b in ([letra_1, letra_2, letra_3]):
            ncm = join_final.iloc[a]["HS6"]
            ncm_val = cont.loc[ncm][b]
            x.append(ncm_val)
        
        total=x[0]+x[1]+x[2]
        act1_pond=x[0]/total
        act2_pond=x[1]/total
        act3_pond=x[2]/total
        join_final.at[a, "letra1_pond"] = act1_pond
        join_final.at[a, "letra2_pond"] = act2_pond
        join_final.at[a, "letra3_pond"] = act3_pond
        print(ncm, x, total, act1_pond, act2_pond, act3_pond)
    return join_final

