

from tqdm import tqdm
import pandas as pd
import numpy as np


def def_asignacion_sec(raw_data, ci = False):
    
    letra_sd = np.nan
    
    letra1 = raw_data.columns.get_loc("letra1") + 1
    letra2 = raw_data.columns.get_loc("letra2") + 1
    letra3 = raw_data.columns.get_loc("letra3") + 1
    letra4 = raw_data.columns.get_loc("letra4") + 1
    letra5 = raw_data.columns.get_loc("letra5") + 1
    letra6 = raw_data.columns.get_loc("letra6") + 1
    
    if ci == False:
        vta_bk = raw_data.columns.get_loc("vta_bk") + 1
        vta_bk2 = raw_data.columns.get_loc("vta_bk2") + 1
        vta_bk3 = raw_data.columns.get_loc("vta_bk3") + 1
        vta_bk4 = raw_data.columns.get_loc("vta_bk4") + 1
        vta_bk5 = raw_data.columns.get_loc("vta_bk5") + 1
        vta_bk6 = raw_data.columns.get_loc("vta_bk6") + 1
    else:
        vta_bk = np.nan
        vta_bk2 = np.nan
        vta_bk3 = np.nan
        vta_bk4 = np.nan
        vta_bk5 = np.nan
        vta_bk6 = np.nan

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
    
    ue_dest_loc =  raw_data.columns.get_loc("ue_dest") + 1
    cuit_loc =  raw_data.columns.get_loc("cuit") + 1
    hs12_loc =  raw_data.columns.get_loc("HS6_d12") + 1
    valor_loc =  raw_data.columns.get_loc("valor") + 1


    dictionary_list = []

    for a in tqdm(raw_data.itertuples()):
        for b, c, d, e in zip([letra1, letra2, letra3,letra4, letra5, letra6],
                              [vta_bk, vta_bk2, vta_bk3,vta_bk4, vta_bk5, vta_bk6],
                              [vta_sec, vta_sec2, vta_sec3, vta_sec4, vta_sec5, vta_sec6],
                              [letra1_pond, letra2_pond, letra3_pond, letra4_pond, letra5_pond, letra6_pond]):

            if a[b] == "G":
                if ci == False:
                    if a[c] == 0:
                        letra_sd = a[b]
                    elif a[d] ==0:
                        letra_sd = "CONS"
                    else:
                       letra_sd = None #np.nan #asigno por probabilidad

            else:
                letra_sd = a[b]

            values = {'cuit': a[cuit_loc],
                      "hs6_d12": a[hs12_loc],
                      "valor_pond": a[valor_loc] * a[e],
                      "si": a[letra1],
                      "sd": letra_sd,
                      "ue_dest": a[ue_dest_loc]}

            dictionary_list.append(values)

    for_fill = pd.DataFrame.from_dict(dictionary_list)

    for_fill["sd"] = np.where(for_fill["hs6_d12"].str.startswith("8703"), 
                              "CONS", for_fill["sd"] )
    
    for_fill["si"] = np.where(for_fill["hs6_d12"].str.startswith(("8702", "8703", "8704")), 
                              "G", for_fill["si"] )

    return for_fill
    
def def_asignacion_prob(prob):
    
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

def def_asignacion_picks(bk_picks, mectra_pond):
    cols = ["cuit", "HS6_d12", "valor", "ue_dest", "letra1"]
    lista= []
    for impo_i in bk_picks[cols].itertuples():
        for coef in mectra_pond.itertuples():
            dic = {"cuit": impo_i[1],
                   "hs6_d12": impo_i[2],
                   "valor_pond": impo_i[3]*coef[2],
                   "si": impo_i[5],
                   "sd": coef[1],
                   "ue_dest": impo_i[4]
                   }
            lista.append(dic)   
    asign_pre_matriz_pick = pd.DataFrame.from_dict(lista)
    return asign_pre_matriz_pick                   
    
def to_matriz(matriz_sisd_final, ci = False):
    #probar con index = hs6
    # matriz_sisd_final = asign_pre_matriz_ci
    
    z = pd.pivot_table(matriz_sisd_final, values='valor_pond', index=['si'], columns=['sd'], aggfunc=np.sum, fill_value=0) 
    cols=list(z.columns.values)
    cols.pop(cols.index("CONS"))
    z=z[cols+["CONS"]] #ubicacion del consumo ultima colummna
    
    if ci == False:
        z= z.append(pd.Series(name='P')) #imputacion de Q
        z= z.replace(np.nan,0)
        
        # z.insert(29, "G", 0) # AGREGO G !!!
        
    elif ci == True:
        z.insert(29, "G", 0)
        z.insert(44, "Q", 0)

   
    
    z= z.append(pd.Series(name='Q')) #imputacion de P
    z= z.replace(np.nan,0)
    
    z= z.append(pd.Series(name='CONS')) #imputacion de CONS
    z= z.replace(np.nan,0)
    
    return z
