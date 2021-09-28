# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:06:12 2021

@author: igalk
"""

import os 
os.getcwd()

import pandas as pd
import numpy as np
import re


# def carga_de_bases(x):
#     impo_17 = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/IMPO_2017.csv", sep=";")
#     clae = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/clae_nombre.csv")
#     print ("hola")
#     return impo_17
# #     comercio = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/comercio_clae.xlsx")
# #     cuit_clae = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/cuit 2017 impo_con_actividad.csv")
# #     bec = pd.read_excel( "/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/HS2012-17-BEC5 -- 08 Nov 2018.xlsx", sheet_name= "HS17BEC5" )
# #     #parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# #     #transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")
# #     bec_to_clae = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/bec_to_clae.xlsx")
# #     return impo_17


def predo_impo_17(impo_17):
    impo_17['FOB'] = impo_17['FOB'].str.replace(",", ".")
    impo_17['FOB'] = impo_17['FOB'].astype(float)
    impo_17['CIF'] = impo_17['CIF'].str.replace(",", ".")
    impo_17['CIF'] = impo_17['CIF'].astype(float)
    impo_17.drop("FOB", axis=1, inplace=True)
    return impo_17.rename(columns = {"POSIC_SIM" : "HS6", 'CIF': "valor"} , inplace = True)


def predo_impo_12d(impo_d12, ncm_desc):

    impo_d12.rename(columns = {'ANYO':"anio", 'POSIC_SIM':"HS6_d12", 
                               'CIF':"valor", "CUIT_IMPOR":"cuit",
                               "KILOS": "kilos", "DESTINAC": "destinacion",  "DEST": "dest_cod",
                               "UMED_ESTAD": "uni_est", "CANT_UNEST": "cant_est",
                               "UMED_DECL": "uni_decl", "CANT_DECL": "cant_decl"}, inplace=True)
    impo_d12 = impo_d12[[ "cuit", "NOMBRE", "HS6_d12", "destinacion", "dest_cod", "valor", "kilos", "uni_est", "cant_est", "uni_decl", "cant_decl"]]
    impo_d12["HS6"]= impo_d12["HS6_d12"].str.slice(0,6).astype(int)
    impo_d12["cuit"] =impo_d12["cuit"].astype(str)
    
    impo_d12 = pd.merge(impo_d12, ncm_desc[["Descripción Completa", "Posición"]], left_on="HS6_d12", right_on="Posición", how="left").drop(["Posición"], axis=1).rename(columns = {"Descripción Completa": "descripcion"})
    
    return impo_d12


def predo_sectores_nombres(clae):
    letras_np = pd.unique(clae['letra'])
    letras_np= np.delete(letras_np, 0)
    letras_np= letras_np[~pd.isnull(letras_np)]
    letras = pd.merge(pd.DataFrame({'letra': letras_np}), clae[['letra', 'letra_desc']],
                      how = "left", left_on = "letra", right_on = "letra")
    letras.drop_duplicates(subset = 'letra', inplace = True)
    letras = pd.DataFrame(letras)
    
    cons = pd.DataFrame([{"letra": "CONS", "letra_desc": "CONSUMO"}] )
    letras  = pd.concat([letras, cons] , axis =0)
    
    return letras


def predo_comercio(comercio, clae):
    comercio.rename(columns = { 'Unnamed: 2': "clae6" , "G": "clae3", 
                           'COMERCIO   AL   POR   MAYOR   Y   AL   POR   MENOR;   REPARACIÓN   DE   VEHÍCULOS AUTOMOTORES Y MOTOCICLETAS' : "clae6_desc",
                            'Venta de vehículos': "vta_vehiculos",
                            'Vende BK': "vta_bk",
                            'Vende mayormente BK a otros sectores productivos o CF?': "vta_sec"}  , inplace = True )
    comercio["clae3"].fillna(comercio['Unnamed: 1'], inplace = True)
    comercio["clae3"].fillna( method = "ffill", inplace = True)
    del comercio['Unnamed: 1']
    comercio.drop(comercio.iloc[:,6: ], axis= 1, inplace= True) 
    
    comercio_reclasificado =  pd.merge( left =clae[clae["letra"] =="G"][["letra", "clae6", "clae6_desc"]], 
                                       right = comercio.drop(["clae6_desc", "clae3"], axis = 1), left_on="clae6", right_on = "clae6", how="left" ) 
    
    comercio_reclasificado["clae6"] = comercio_reclasificado["clae6"].astype(str)
    return comercio_reclasificado

def predo_cuit_clae(cuit_clae , clae):
    
    cuit_clae = cuit_clae[cuit_clae["Numero_actividad_cuit"]<=6].drop(["Clae6_desc","Fecha_actividad", "Cantidad_actividades_cuit"], axis =1)
    cuit_clae = cuit_clae.drop_duplicates()
    
    #procesamiento de los 6 digitos
    cuit_clae6 = cuit_clae.set_index(["CUIT", "Numero_actividad_cuit"], append=True)
    cuit_clae6 = cuit_clae6.unstack().droplevel(0, axis=1)
    cuit_clae6 = cuit_clae6.groupby("CUIT").sum().astype(int)
    cuit_clae6.columns = ["actividad1","actividad2","actividad3","actividad4","actividad5","actividad6"]
    
    #garantizo que todos los cuit tengan actividaes
    cuit_clae6.loc[cuit_clae6['actividad6']<100, ['actividad6']] = 0
    cuit_clae6.loc[cuit_clae6['actividad5']<100, ['actividad5']] = cuit_clae6['actividad6']
    cuit_clae6.loc[cuit_clae6['actividad4']<100, ['actividad4']] = cuit_clae6['actividad5']
    cuit_clae6.loc[cuit_clae6['actividad3']<100, ['actividad3']] = cuit_clae6['actividad4']
    cuit_clae6.loc[cuit_clae6['actividad2']<100, ['actividad2']] = cuit_clae6['actividad3']
    cuit_clae6.loc[cuit_clae6['actividad1']<100, ['actividad1']] = cuit_clae6['actividad2']
    
    #relleno los que tienen solo act 1
    cuit_clae6["suma"] = cuit_clae6.loc[:,["actividad2","actividad3","actividad4","actividad5","actividad6"]].sum(axis=1)
    
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad2']] = cuit_clae6['actividad1']
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad3']] = cuit_clae6['actividad1']    
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad4']] = cuit_clae6['actividad1']    
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad5']] = cuit_clae6['actividad1']
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad6']] = cuit_clae6['actividad1']
    
    #relleno los que tienen hasta act 2
    cuit_clae6["suma"] = cuit_clae6.loc[:,["actividad3","actividad4","actividad5","actividad6"]].sum(axis=1)    
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad3']] = cuit_clae6['actividad1']
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad5']] = cuit_clae6['actividad1']
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad4']] = cuit_clae6['actividad2']
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad6']] = cuit_clae6['actividad2']
    
    #relleno los que tienen hasta act 3
    cuit_clae6["suma"] = cuit_clae6.loc[:,["actividad4","actividad5","actividad6"]].sum(axis=1)    
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad4']] = cuit_clae6['actividad1']
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad5']] = cuit_clae6['actividad2']
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad6']] = cuit_clae6['actividad3']
    
    #relleno los que tienen hasta act 4
    cuit_clae6["suma"] = cuit_clae6.loc[:,["actividad5","actividad6"]].sum(axis=1)    
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad5']] = cuit_clae6['actividad1']
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad6']] = cuit_clae6['actividad2']
    
    #relleno los que tienen hasta act 5
    cuit_clae6["suma"] = cuit_clae6.loc[:,["actividad6"]].sum(axis=1)    
    cuit_clae6.loc[cuit_clae6['suma']<1, ['actividad6']] = cuit_clae6['actividad1']
    
    cuit_clae6 = cuit_clae6.drop(["suma"], axis=1)
    
    #procesamiento a letras
    clae["clae6"] = clae["clae6"].astype("Int64").astype(str)
    cuit_clae6 = cuit_clae6.reset_index().astype(str)
    
    letra1 = pd.merge(cuit_clae6[["CUIT","actividad1"]].astype(str), clae[["clae6", "letra"]], left_on="actividad1", right_on="clae6", how="left").drop(["actividad1", "clae6"], axis=1).rename(columns={"letra": "letra1"})
    letra2 = pd.merge(cuit_clae6[["CUIT","actividad2"]].astype(str), clae[["clae6", "letra"]], left_on="actividad2", right_on="clae6", how="left").drop(["actividad2", "clae6"], axis=1).rename(columns={"letra": "letra2"})
    letra3 = pd.merge(cuit_clae6[["CUIT","actividad3"]].astype(str), clae[["clae6", "letra"]], left_on="actividad3", right_on="clae6", how="left").drop(["actividad3", "clae6"], axis=1).rename(columns={"letra": "letra3"})
    letra4 = pd.merge(cuit_clae6[["CUIT","actividad4"]].astype(str), clae[["clae6", "letra"]], left_on="actividad4", right_on="clae6", how="left").drop(["actividad4", "clae6"], axis=1).rename(columns={"letra": "letra4"})
    letra5 = pd.merge(cuit_clae6[["CUIT","actividad5"]].astype(str), clae[["clae6", "letra"]], left_on="actividad5", right_on="clae6", how="left").drop(["actividad5", "clae6"], axis=1).rename(columns={"letra": "letra5"})
    letra6 = pd.merge(cuit_clae6[["CUIT","actividad6"]].astype(str), clae[["clae6", "letra"]], left_on="actividad6", right_on="clae6", how="left").drop(["actividad6", "clae6"], axis=1).rename(columns={"letra": "letra6"})
    
    
    cuit_clae6 = cuit_clae6.merge(letra1, left_on="CUIT", right_on="CUIT")
    cuit_clae6 = cuit_clae6.merge(letra2, left_on="CUIT", right_on="CUIT")
    cuit_clae6 = cuit_clae6.merge(letra3, left_on="CUIT", right_on="CUIT")
    cuit_clae6 = cuit_clae6.merge(letra4, left_on="CUIT", right_on="CUIT")
    cuit_clae6 = cuit_clae6.merge(letra5, left_on="CUIT", right_on="CUIT")
    cuit_clae6 = cuit_clae6.merge(letra6, left_on="CUIT", right_on="CUIT")
    
    return cuit_clae6 
    
     
def predo_bec_bk(bec, bec_to_clae):
    bec_cap = bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]
    
    #partes y accesorios dentro de start with cap
    partes_accesorios  = bec_cap[bec_cap["HS6Desc"].str.contains("part|acces")]   
    # partes_accesorios["BEC5EndUse"].value_counts()

    # filtro bienes de capital
    bec_bk = bec_cap.loc[~bec_cap["HS6"].isin( partes_accesorios["HS6"])]
    
    filtro = [ "HS4", "HS4Desc", "HS6", "HS6Desc", "BEC5Category", "BEC5EndUse"]
    
    bec_bk = bec_bk[filtro]
    
    return bec_bk


def predo_stp(dic_stp ):

    dic_stp.columns = ["NCM", "desc", "ciiu", "desc_gral", "utilizacion", "demanda"]
    dic_stp.dropna(thresh = 3, inplace= True)
    agro_stp = dic_stp[dic_stp["demanda"].str.contains("agrí", case = False)]

    return dic_stp

def predo_ncm(ncm_sector):
    ncm_sector["hs6"] = ncm_sector["hs6"].astype(int)
    return ncm_sector

def def_join_impo_clae(impo_anyo_12d, cuit_empresas):
    impo_clae = pd.merge(impo_anyo_12d, cuit_empresas, left_on = "cuit", right_on = "CUIT", how = "right")
    impo_clae.drop(["CUIT"], axis=1, inplace = True)
       
    return impo_clae

def def_join_impo_clae_bec(join_impo_clae, bec_bk):
    impo_anyo_12d_bec_bk = pd.merge(join_impo_clae, bec_bk, how= "left" , left_on = "HS6", right_on= "HS6" )

    # filtramos las impos que no mergearon (no arrancan con CAP)
    impo_anyo_12d_bec_bk = impo_anyo_12d_bec_bk[impo_anyo_12d_bec_bk["HS4"].notnull()]
    
    impo_anyo_12d_bec_bk["ue_dest"] = ""
    return impo_anyo_12d_bec_bk

    

def def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio):
    
    comercio = comercio.drop(["vta_vehiculos"], axis=1)
    
    comercio2 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk2", "vta_sec": "vta_sec2"})

    comercio3 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk3", "vta_sec": "vta_sec3"})    

    comercio4 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk4", "vta_sec": "vta_sec4"})

    comercio5 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk5", "vta_sec": "vta_sec5"})

    comercio6 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk6", "vta_sec": "vta_sec6"})

    # join de la matriz con el sector comercio
    ## Comercio 1
    impo17_bec_complete = pd.merge(join_impo_clae_bec_bk, comercio.drop(["letra", "clae6_desc"], axis = 1), 
                             how = "left", left_on = "actividad1", right_on = "clae6")
    
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 

    ## Comercio 2
    impo17_bec_complete = pd.merge(impo17_bec_complete, comercio2, 
                                   how = "left", left_on = "actividad2", right_on = "clae6")
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 

    ## Comercio 3
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio3 , 
                          how = "left", left_on = "actividad3", right_on = "clae6")
    
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 

    ## Comercio 4
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio4 , 
                          how = "left", left_on = "actividad3", right_on = "clae6")
    
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 
    
    ## Comercio 5
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio5 , 
                          how = "left", left_on = "actividad3", right_on = "clae6")
    
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 
    
    ## Comercio 6
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio6 , 
                          how = "left", left_on = "actividad3", right_on = "clae6")
    
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 

    return  impo17_bec_complete   

    
def destinacion_limpio(x):
    if re.search("PARA TRANSF|C/TRANS|P/TRANS|RAF|C/TRNSF|ING.ZF INSUMOS", x)!=None:
        return "C/TR"
    elif re.search("S/TRAN|SIN TRANSF|INGR.ZF BIENES", x)!=None:
        return "S/TR"
    elif re.search("CONS|ING.ZF MERC", x)!=None:
        # return "CONS"
        return "CONS&Otros"
    else: 
        # return "Otro"
        return "CONS&Otros"
    
def metrica(x):
    return (x["valor"] * x["kilos"])/x["cant_decl"]

def mod_z(col: pd.Series, thresh: float=3.5):
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = 0.645* ((col - med_col) / med_abs_dev)
    # mod_z = mod_z[np.abs(mod_z) < thresh]
    return np.abs(mod_z)

