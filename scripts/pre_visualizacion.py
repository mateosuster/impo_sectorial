# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:09:44 2021

@author: mateo
"""

import pandas as pd
import numpy as np


def impo_total(z, industria_2d):
     #filtro industria   
    z_industria = z.loc[:, industria_2d["clae2"]]

    # transformacion a array de np
    indu_np  = z_industria.to_numpy()

    # totales col 
    col_sum  = np.nansum(indu_np , axis=0)
    
    #importaciones totales (ordenadas)
    impo_tot_sec = pd.DataFrame({"Importaciones totales": col_sum/(10**6), "clae_2d": list(industria_2d["clae2"])}, index=industria_2d["desc"])  
    impo_tot_sec.sort_values(by = "Importaciones totales", ascending= False, inplace = True)

    return impo_tot_sec



def impo_comercio_y_propio(z, industria_2d , comercio_2d):
    
    z_industria = z.loc[:, industria_2d["clae2"]]
    indu_np  = z_industria.to_numpy()
    col_sum  = np.nansum(indu_np , axis=0)
    
    #diagonal sobre total col y comercio sobre total
    z_industria_p_diag = z.loc[industria_2d["clae2"], industria_2d["clae2"]]
    diagonal = np.diag(z_industria_p_diag )
    diag_total_col = diagonal/col_sum
    
    z_industria_g = z.loc[comercio_2d["clae2"], industria_2d["clae2"]]
    sum_col_c_g = np.nansum(z_industria_g, axis = 0 )
    g_total_col = sum_col_c_g /col_sum
    
    comercio_y_propio = pd.DataFrame({"Propio": diag_total_col*100 , 'Comercio': g_total_col*100} , index = industria_2d["desc"]).sort_values(by = 'Propio', ascending = False)
    
    return comercio_y_propio



def top_5(matriz_sisd_final, industria_2d, bec, impo_tot_sec):
    
    x = matriz_sisd_final[matriz_sisd_final["sd"].isin(industria_2d["clae2"])].groupby(["hs6", "sd"], as_index=False)['valor_pond'].sum("valor_pond")
    top_5_impo = x.reset_index(drop = True).sort_values(by = ["sd", "valor_pond"],ascending = False)
    top_5_impo  = top_5_impo.groupby(["sd"], as_index = True).head(5)
    top_5_impo  = pd.merge(left=top_5_impo, right=industria_2d, left_on="sd", right_on="clae2", how="left").drop(["sd", "desc"], axis=1)
    top_5_impo  = pd.merge(left=top_5_impo, right=bec[["HS6","HS6Desc"]], left_on="hs6", right_on="HS6", how="left").drop("HS6", axis=1)
    top_5_impo  = pd.merge(top_5_impo  , impo_tot_sec, left_on="clae2", right_on="clae_2d", how = "left")
    top_5_impo["impo_relativa"] = top_5_impo["valor_pond"]/(top_5_impo["Importaciones totales"] * 10**6) 
    
    return top_5_impo


def predo_bce_cambiario(bce_cambiario):
    bce_cambiario.drop(bce_cambiario.columns[16:], axis=1, inplace=True)
    bce_cambiario.rename(columns= {"Años": "anio", "ANEXO": "partida", "Denominación": "sector" }, inplace= True)
    
    #conversion a numeros
    to_num_cols= bce_cambiario.columns[4:]
    bce_cambiario[to_num_cols] = bce_cambiario[to_num_cols].apply(pd.to_numeric,errors='coerce')
    
    #filtro
    bce_cambiario_filter = bce_cambiario[(bce_cambiario["anio"] ==2017) & (bce_cambiario["partida"].isin([7,8,10]) )].drop([ "anio", "partida", "C-V"], axis=1) 
    
    impo_tot_bcra = bce_cambiario_filter.groupby( "sector", as_index = True).sum().sum(axis= 1).reset_index().rename(columns= { 0: "impo_bcra"} )
    
    return impo_tot_bcra 


def merge_bcra(impo_tot_sec, impo_tot_bcra):
    
    impo_tot_sec = impo_tot_sec.copy()
    
    impo_tot_sec.drop("clae_2d", axis=1, inplace= True)
    x1 = impo_tot_sec.loc[["Alimentos", "Bebidas", "Tabaco"]].sum().reset_index(drop=True)
    x2 = impo_tot_sec.loc[["Equipos de automotores", "Equipo de transporte"]].sum().reset_index(drop=True)
    x3 = impo_tot_sec.loc[["Químicos", "Productos de coque", "Caucho y vidrio"]].sum().reset_index(drop=True)
    x4 = impo_tot_sec.loc[["Textiles", "Prendas de vestir", "Productos de cuero"]].sum().reset_index(drop=True)
    x5 = impo_tot_sec.loc[["Productos de papel", "Imprentas"]].sum().reset_index(drop=True)
    x6 = impo_tot_sec.loc[["Maquinaria"]].sum().reset_index(drop=True)
    x7 = impo_tot_sec.loc[["Hierro y acero", "Productos metálicos"]].sum().reset_index(drop=True)
    x8 = impo_tot_sec.loc[["Minerales no metálicos"]].sum().reset_index(drop=True)
    
    
    dic = {"sector": ["Alimentos, Bebidas y Tabaco", "Industria Automotriz",
                      "Industria Química, Caucho y Plástico", 
                      "Industria Textil y Curtidos", 
                      "Industria de Papel, Ediciones e Impresiones" ,
                      "Maquinarias y Equipos", 
                      "Metales Comunes y Elaboración",
                      "Productos Minerales no Metálicos (Cementos, Cerámicos y Otros)"] , 
           "impo_sisd": pd.concat([x1, x2, x3,x4,x5,x6,x7, x8])}
    
    
    comparacion_bcra = pd.merge(pd.DataFrame(dic), impo_tot_bcra, 
                                left_on = "sector", right_on = "sector", how = "left")
    
    comparacion_bcra["sector"] =comparacion_bcra["sector"].str.replace( "\(Cementos, Cerámicos y Otros\)", "")
    comparacion_bcra.sort_values(by = 'impo_sisd', ascending = False, inplace =True)
    return comparacion_bcra


def dic_clae_ciiu(isic, dic_ciiu ):
    
    isic=isic.iloc[:,[0,2]]
    isic.columns= ["HS6", "ISIC"]
    

    dic_clae_ciiu = pd.merge(isic, dic_ciiu , how= "left" , left_on ="ISIC", right_on ="ciiu3_4c").drop("ciiu3_4c", axis =1)
    dic_clae_ciiu["clae6"] = dic_clae_ciiu["clae6"].astype(str).str.slice(0,2)
    
    return dic_clae_ciiu
    
    
def join_sisd_ciiu(join_impo_clae_bec_bk, dic_clae_ciiu ,impo_tot_sec, industria_2d):
    
    impo_ciiu =pd.merge(join_impo_clae_bec_bk[["HS6", "valor"]], dic_clae_ciiu , how = "left" ,right_on="HS6", left_on="HS6")
    impo_ciiu_indu = impo_ciiu[impo_ciiu["clae6"].isin(industria_2d["clae2"])]
    impo_ciiu_letra =impo_ciiu_indu.groupby("clae6")["valor"].apply(lambda x: x.sum()/10**6).reset_index()
    
    comparacion_ciiu = pd.merge(impo_tot_sec.reset_index(), impo_ciiu_letra, how="right", left_on = "clae_2d", right_on = "clae6" ).drop("clae6", axis = 1).rename(columns = { "valor":"impo_ciiu", "Importaciones totales": "impo_sisd" }  ).sort_values(by = "impo_sisd", ascending = False)
    
    return comparacion_ciiu
    
