# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:46:35 2021

@author: igalk
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def def_contingencia(join_impo_clae_bec_bk_comercio, todos):
    # join_impo_clae_bec_bk_comercio, todos = datos_bk_comercio, datos
    # impo_by_product_1 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra1" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra1": "letra"})
    # impo_by_product_2 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra2" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra2": "letra"})
    # impo_by_product_3 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra3" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra3": "letra"})
    # impo_by_product_4 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra4" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra4": "letra"})
    # impo_by_product_5 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra5" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra5": "letra"})
    # impo_by_product_6 = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", "letra6" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra6": "letra"})

    for i in range(1,7):
        letra_i ="letra"+str(i)
        globals()["impo_by_product_%s" % i] = join_impo_clae_bec_bk_comercio.groupby(["HS6_d12", letra_i], as_index=False).agg({'valor': sum}).rename(columns={letra_i: "letra"})

    impo_by_product = impo_by_product_1.append([impo_by_product_2,impo_by_product_3,impo_by_product_4,impo_by_product_5,impo_by_product_6], ignore_index=True)
    contingencia = pd.pivot_table(impo_by_product, values='valor', index=['HS6_d12'], columns=['letra'], aggfunc=np.sum, fill_value=0)

    #armo una tabla para tener (ncm, expo) con expo = exponente al que elevar la suma del ncm
    ncm_expo = pd.DataFrame(todos["HS6_d12"].value_counts()).reset_index().set_index("index")

    dictionary_list = []

    for i in list(contingencia.itertuples()):
        ncm = i[0]
        expo = ncm_expo.loc[ncm,"HS6_d12"]
        acts = np.array(i[1:])
        acts_pow = pow(acts,(2+np.log10(expo)))

        #guardo la fila en un nuevo df
        values = [ncm, *acts_pow] #el asterisco hace unpack list
        dictionary_list.append(values)

    table = pd.DataFrame.from_dict(dictionary_list)
    table =  table.set_index(table.columns[0])
    table.columns = contingencia.columns


    return table


def def_ponderaciones(impo,cont, ci = False):
    # impo,cont,ci = datos_bk_comercio, tabla_contingencia , False

    join_final = impo.copy()
    
    join_final["letra1_pond"] = np.nan
    join_final["letra2_pond"] = np.nan
    join_final["letra3_pond"] = np.nan
    join_final["letra4_pond"] = np.nan
    join_final["letra5_pond"] = np.nan
    join_final["letra6_pond"] = np.nan
    
    letra1 = join_final.columns.get_loc("letra1") + 1
    letra2 = join_final.columns.get_loc("letra2") + 1
    letra3 = join_final.columns.get_loc("letra3") + 1
    letra4 = join_final.columns.get_loc("letra4") + 1
    letra5 = join_final.columns.get_loc("letra5") + 1
    letra6 = join_final.columns.get_loc("letra6") + 1
    HSd12 = join_final.columns.get_loc("HS6_d12") + 1

    dictionary_list = []
    for a in tqdm(join_final.itertuples()):
        letra_1 = a[letra1]
        letra_2 = a[letra2]
        letra_3 = a[letra3]
        letra_4 = a[letra4]
        letra_5 = a[letra5]
        letra_6 = a[letra6]

        ncm_val = cont.loc[a[HSd12]]

        total = ncm_val[letra_1] + ncm_val[letra_2] + ncm_val[letra_3] + ncm_val[letra_4] + ncm_val[letra_5] + ncm_val[letra_6]
        act1_pond = ncm_val[letra_1] / total
        act2_pond = ncm_val[letra_2] / total
        act3_pond = ncm_val[letra_3] / total
        act4_pond = ncm_val[letra_4] / total
        act5_pond = ncm_val[letra_5] / total
        act6_pond = ncm_val[letra_6] / total

        if ci == False:
            values = [*a[1:45], act1_pond, act2_pond, act3_pond, act4_pond, act5_pond, act6_pond] #el asterisco sirve para seleccionando los elementos de la lista (tupla) a
        elif ci == True:
            values = [*a[1:39], act1_pond, act2_pond, act3_pond, act4_pond, act5_pond, act6_pond] #el asterisco sirve para seleccionando los elementos de la lista (tupla) a
        dictionary_list.append(values)

    rtn = pd.DataFrame.from_dict(dictionary_list)
    rtn.columns = join_final.columns

    return rtn

