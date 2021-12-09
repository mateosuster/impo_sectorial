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
from scipy.stats import median_abs_deviation , zscore



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

def predo_impo_all(impo_d12, name_file):
    impo_d12["anyo"] = impo_d12["FECH_OFIC"].str.slice(0,4)

    #impo_d12 = impo_d12[['anyo','CUIT_IMPOR', 'NOMBRE']]
    impo_d12 = impo_d12[impo_d12["anyo"]=="2017"]
    #impo_d12 = impo_d12.drop_duplicates()

    impo_d12.to_csv("../data/"+name_file, index = False)
    return impo_d12



def predo_impo_17(impo_17):
    impo_17['FOB'] = impo_17['FOB'].str.replace(",", ".")
    impo_17['FOB'] = impo_17['FOB'].astype(float)
    impo_17['CIF'] = impo_17['CIF'].str.replace(",", ".")
    impo_17['CIF'] = impo_17['CIF'].astype(float)
    impo_17.drop("FOB", axis=1, inplace=True)
    return impo_17.rename(columns = {"POSIC_SIM" : "HS6", 'CIF': "valor"} , inplace = True)


def predo_impo_12d(impo_d12, ncm_desc):
    impo_d12["ANYO"] = impo_d12["FECH_OFIC"].str.slice(0,4) 
    impo_d12.rename(columns = {'ANYO':"anio", 'POSIC_SIM':"HS6_d12", 
                               'CIF':"valor", "CUIT_IMPOR":"cuit",
                               "KILOS": "kilos", "DESTINAC": "destinacion",  "DEST": "dest_cod",
                               "UMED_ESTAD": "uni_est", "CANT_UNEST": "cant_est",
                               "UMED_DECL": "uni_decl", "CANT_DECL": "cant_decl"}, inplace=True)
    impo_d12 = impo_d12[[ "cuit", "NOMBRE", "HS6_d12", "destinacion", "dest_cod", "valor", "kilos", "uni_est", "cant_est", "uni_decl", "cant_decl"]]
    impo_d12["HS6"]= impo_d12["HS6_d12"].str.slice(0,6).astype(int)
    impo_d12["cuit"] =impo_d12["cuit"].astype(str)

    # ncm_desc_copy = ncm_desc.rename(columns={'HS_12d': "Posición",
    #                          'hs6_d12_desc': "Descripción Completa"}, inplace= False).copy()

    # impo_d12 = pd.merge(impo_d12, ncm_desc_copy[["Descripción Completa", "Posición"]], left_on="HS6_d12", right_on="Posición", how="left").drop(["Posición"], axis=1).rename(columns = {"Descripción Completa": "descripcion"})
    impo_d12 = pd.merge(impo_d12, ncm_desc, left_on="HS6_d12", right_on="HS_12d", how="left").drop(["HS_12d"], axis=1).rename(columns = {"hs6_d12_desc": "descripcion"})
    
    return impo_d12

def predo_ncm12_desc(ncm12_desc ):
    ncm12_desc = ncm12_desc[["POSICION", "DESCRIPCIO"]]
    ncm12_desc.rename(columns = {"POSICION": "HS_12d", "DESCRIPCIO":"hs6_d12_desc"}, inplace = True) 
    ncm12_desc_split = pd.concat([ncm12_desc.iloc[:,0], pd.DataFrame(ncm12_desc['hs6_d12_desc'].str.split('//', expand=True))], axis=1)
    dic = {"ncm_desc": ncm12_desc, "ncm_split": ncm12_desc_split }
    return dic

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
    
    comercio_reclasificado["clae6"] = comercio_reclasificado["clae6"].astype(int)
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
    
     
def predo_bec_bk(bec):#, bec_to_clae):
    bec_cap = bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]
    filtro = [ "HS4", "HS4Desc", "HS6", "HS6Desc", "BEC5Category", "BEC5EndUse"]
    #partes y accesorios dentro de start with cap
    # partes_accesorios  = bec_cap[bec_cap["HS6Desc"].str.contains("part|acces")]   
    # partes_accesorios["BEC5EndUse"].value_counts()
    # filtro bienes de capital
    # bec_bk = bec_cap.loc[~bec_cap["HS6"].isin( partes_accesorios["HS6"])]
    # bec_bk = bec_bk[filtro]
    bec_cap = bec_cap[filtro]
    
    return bec_cap


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
    impo_clae["dest_clean"] = impo_clae["destinacion"].apply(lambda x: destinacion_limpio(x))

    return impo_clae

def def_join_impo_clae_bec(join_impo_clae, bec):
    impo_bec = pd.merge(join_impo_clae, bec[["HS6", "BEC5EndUse" ]], how= "left" , left_on = "HS6", right_on= "HS6" )
    return impo_bec

def def_join_impo_clae_bec_bk(join_impo_clae, bec_bk):
    impo_anyo_12d_bec_bk = pd.merge(join_impo_clae, bec_bk, how= "left" , left_on = "HS6", right_on= "HS6" )

    # filtramos las impos que no mergearon (no arrancan con CAP)
    impo_anyo_12d_bec_bk = impo_anyo_12d_bec_bk[impo_anyo_12d_bec_bk["HS4"].notnull()]
    
    impo_anyo_12d_bec_bk["ue_dest"] = ""
    return impo_anyo_12d_bec_bk


def def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio, ci = False):
   
    
    if ci == False:    
        #comercio = comercio.drop(["vta_vehiculos"], axis=1)
        comercio2 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk2", "vta_sec": "vta_sec2"})
        comercio3 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk3", "vta_sec": "vta_sec3"})    
        comercio4 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk4", "vta_sec": "vta_sec4"})
        comercio5 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk5", "vta_sec": "vta_sec5"})
        comercio6 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_bk": "vta_bk6", "vta_sec": "vta_sec6"})
    elif ci == True:
        comercio2 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_sec": "vta_sec2"})
        comercio3 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_sec": "vta_sec3"})    
        comercio4 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_sec": "vta_sec4"})
        comercio5 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_sec": "vta_sec5"})
        comercio6 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = {"vta_sec": "vta_sec6"})
            
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

    
# =============================================================================
# clasificación via destinación
# =============================================================================

    
def metrica(x):
    return (x["valor"] * x["kilos"])/x["cant_decl"]

def mod_z(col: pd.Series, thresh: float=3.5):
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = 0.645* ((col - med_col) / med_abs_dev)
    # mod_z = mod_z[np.abs(mod_z) < thresh]
    return np.abs(mod_z)


# =============================================================================
# preprocesamiento data modelo xgboost
# =============================================================================
def str_a_num(df):
  for col in df:
    original = np.sort(df[col].unique())
    reemplazo = list(range(len(original)))
    mapa = dict(zip(original, reemplazo))
    df.loc[:,col] = df.loc[:,col].replace(mapa)
  return(df)

# =============================================================================
# ordenamiento de los datos de modelo
# =============================================================================
def predo_datamodel(data_predichos,datos_clasificados):
    data_predichos["ue_dest"] =np.where(data_predichos["bk_dummy"]== 1, "BK" , "CI")
    data_predichos.drop("bk_dummy", 1, inplace=True)
    
    datos_clasificados = datos_clasificados [datos_clasificados ["ue_dest"] != "?" ]
    
    datos = pd.concat([data_predichos, datos_clasificados], 0)
    return datos


def asignacion_stp_BK(datos, dic_stp): # input: all data; output: BK
    datos_bk = datos[datos["ue_dest"]== "BK"]
    
    ncm_trans = [
                870421, 870431, #pick-ups confirmado por Maito
                 870422, 870423, 870490, 870432, 
                 870210, 870210,870290 
                 # 870310, 870321, 870322, 870323, 870324, 870331, 870332,
                 # 870333, 870390 # mandar a consumo
                 ]

    data_trans = datos_bk[datos_bk["HS6"].isin(ncm_trans)].reset_index(drop = True)
    
    ncm_agro = dic_stp[dic_stp["demanda"].str.contains("agríc", case =False)]["NCM"]
    data_agro = datos_bk[datos_bk["HS6"].isin(ncm_agro)].reset_index(drop = True)
    
    datos_bk_filtro = datos_bk[~(datos_bk["HS6"].isin(ncm_trans)) & ~(datos_bk["HS6"].isin(ncm_agro))]
    
    
    letras = ["letra1", "letra2", "letra3","letra4", "letra5", "letra6"]
    for letra in letras:
        data_trans[letra] = data_trans[letra].replace(regex=[r'^D.*$'],value="I_60") #con regex, buscar que empiece con D_ y poner I_60
        data_trans[letra] = data_trans[letra].replace(["G","K", "J"] , "I_60")
    
    for letra in letras:
        data_agro[letra] = data_agro[letra].replace(regex=[r'^D.*$'],value="A") #con regex, buscar que empiece con D_ y poner A
        data_agro[letra] = data_agro[letra].replace(["G","K", "J"] , "A")
    
    datos_bk = pd.concat([datos_bk_filtro, data_trans, data_agro], axis=0)

    return datos_bk

def filtro_ci(datos):
    datos_ci= datos[datos["ue_dest"]== "CI"]
    return datos_ci

def letra_nn(datos_bk):
    dic = []
    dic_list = []
    
    letra1 = datos_bk.columns.get_loc("letra1") + 1
    letra2 = datos_bk.columns.get_loc("letra2") + 1
    letra3 = datos_bk.columns.get_loc("letra3") + 1
    letra4 = datos_bk.columns.get_loc("letra4") + 1
    letra5 = datos_bk.columns.get_loc("letra5") + 1
    letra6 = datos_bk.columns.get_loc("letra6") + 1
    
    actividad1 = datos_bk.columns.get_loc("actividad1") + 1
    actividad2 = datos_bk.columns.get_loc("actividad2") + 1
    actividad3 = datos_bk.columns.get_loc("actividad3") + 1
    actividad4 = datos_bk.columns.get_loc("actividad4") + 1
    actividad5 = datos_bk.columns.get_loc("actividad5") + 1
    actividad6 = datos_bk.columns.get_loc("actividad6") + 1
    
    
    for x in datos_bk.itertuples():
        dic = []
        for letra, act, letra_name in zip([letra1, letra2, letra3,letra4, letra5, letra6],
                                     [actividad1, actividad2, actividad3,actividad4, actividad5, actividad6],
                                     ["letra1", "letra2", "letra3","letra4", "letra5", "letra6"]):
            
            if x[letra] in [ "H"]:
                new_letra= x[letra] +"_"+ str(x[act])[0:3] 
            elif x[letra] == "D": 
                if str(x[act]) == "29_30_31_32_33":
                    new_letra =  "D_29_30_31_32_33"
                else:
                    new_letra= x[letra] +"_"+ str(x[act])[0:2]
            elif x[letra] in ["I"]:
                new_letra= x[letra] +"_"+ str(x[act])[0:2] 
            #     if str(x[act]) == "64":
            #         new_letra= x[letra] +"_"+ str(x[act])[0:2]
            #     else: 
            #         new_letra = "I_60_61_62_63"
            else:
                new_letra= x[letra] 
                
            # dic[letra_name] = new_letra
            dic.append(new_letra)
            
        dic_list.append(dic)    
        
    letras_trans = pd.DataFrame.from_dict(dic_list) 
    letras_trans.columns = ["letra1","letra2","letra3","letra4", "letra5", "letra6"]
    return letras_trans
 
   
 
 # CIIU 
def predo_dic_ciiu(dic_ciiu):
    dic_ciiu["ciiu3_4c"] = dic_ciiu["ciiu3_4c"].astype(str)
    return dic_ciiu



def predo_ciiu(clae_to_ciiu, dic_ciiu):
    # Join ciiu digitos con letra (posee clae para hacer el join)
    clae_to_ciiu.loc[clae_to_ciiu["ciiu3_4c"].isnull(),"ciiu3_4c"  ] = 4539
    ciiu_dig_let = pd.merge(dic_ciiu[["ciiu3_4c", "ciiu3_letra"]], clae_to_ciiu.drop("clae6_desc", 1), left_on = "ciiu3_4c", right_on = "ciiu3_4c" , how = "inner") #"ciiu3_4c_desc"
    
    #convierto a string los digitos del ciiu
    ciiu_dig_let["ciiu3_4c"] = ciiu_dig_let["ciiu3_4c"].astype(int).astype(str)
    
    #filtro el caso problematico del clae    
    ciiu_dig_let = ciiu_dig_let[ciiu_dig_let["clae6"] !=332000] 
    
    #agrego los clae faltantes
    claes_faltantes = pd.DataFrame({'clae6': [204000, 523032, 462110, 332000], 'ciiu3_letra': ["D", "I", "G", "D"  ] , 
                                    # "ciiu3_4c_desc" : ["", "", "", ""],
                                    'ciiu3_4c': ["2429", "6350", "5121", "29_30_31_32_33"]})    
    ciiu_dig_let = pd.concat([ciiu_dig_let, claes_faltantes ], axis = 0)
    return ciiu_dig_let


def diccionario_especial(datos_bk,ciiu_dig_let):
    # conversion CLAE a CIIU (codigo para funcion)
    datos_bk_a_ciiu= datos_bk[[ "actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6" ,
                               "letra1","letra2","letra3","letra4", "letra5", "letra6"]].copy()
    
    # ESTE ES EL POSTA
    for clae_i, ciiu_name, letra_name in zip(["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6"],
                               ["ciiu_act1", "ciiu_act2", "ciiu_act3", "ciiu_act4", "ciiu_act5", "ciiu_act6"],
                               ["ciiu_letra1","ciiu_letra2","ciiu_letra3","ciiu_letra4", "ciiu_letra5", "ciiu_letra6"]):
        
        # ciiu_data.rename(columns = {"ciiu3_4c": ciiu_name })
        datos_bk_a_ciiu= pd.merge(datos_bk_a_ciiu , ciiu_dig_let, how = "left" ,left_on = clae_i, right_on= "clae6" ).drop("clae6", 1)
        datos_bk_a_ciiu.rename(columns = {"ciiu3_4c": ciiu_name, "ciiu3_letra": letra_name  }, inplace = True)
    
    # datos_bk_a_ciiu.isnull().values.any()
    # datos_bk_a_ciiu.isnull().sum()
    

    for clae_i, clae_letra_i, ciiu_i, ciiu_letra_i, dic_i, dic_letra_i in zip(["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6"],
                                                          ["letra1","letra2","letra3","letra4", "letra5", "letra6"],                     
                                                       ["ciiu_act1", "ciiu_act2", "ciiu_act3", "ciiu_act4", "ciiu_act5", "ciiu_act6"],
                                                       ["ciiu_letra1","ciiu_letra2","ciiu_letra3","ciiu_letra4", "ciiu_letra5", "ciiu_letra6"] ,
                                                       ["dic_act1", "dic_act2", "dic_act3", "dic_act4", "dic_act5", "dic_act6"],
                                                       ["dic_letra1","dic_letra2","dic_letra3","dic_letra4", "dic_letra5", "dic_letra6"]):
    
        datos_bk_a_ciiu[dic_i] = np.where(datos_bk_a_ciiu[clae_letra_i]=="G", datos_bk_a_ciiu[clae_i] , datos_bk_a_ciiu[ciiu_i])
        datos_bk_a_ciiu[dic_letra_i] = np.where(datos_bk_a_ciiu[clae_letra_i]=="G", datos_bk_a_ciiu[clae_letra_i] , datos_bk_a_ciiu[ciiu_letra_i])
    
    # datos_bk_a_ciiu.isnull().values.any()
    # datos_bk_a_ciiu.to_csv("../data/resultados/datos_bk_a_ciiu.csv")


    # tiro las columnas que vamos a unir
    datos_bk.drop( ["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6",
             "letra1","letra2","letra3","letra4", "letra5", "letra6"], axis = 1, inplace = True)     
    datos_bk_a_ciiu.drop(["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6",
                          "letra1","letra2","letra3","letra4", "letra5", "letra6" ],axis = 1, inplace = True)

    # renombro columnas
    datos_bk_a_ciiu.rename(columns = {"dic_act1": "actividad1", "dic_act2" :"actividad2","dic_act3" :"actividad3", 
                                      "dic_act4" : "actividad4","dic_act5": "actividad5","dic_act6": "actividad6",
                                      "dic_letra1": "letra1","dic_letra2":"letra2","dic_letra3":"letra3",
                                      "dic_letra4":"letra4", "dic_letra5":"letra5", "dic_letra6":"letra6"
                                      }, inplace = True)
    #selecciono
    datos_bk_a_ciiu = datos_bk_a_ciiu[["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6",
             "letra1","letra2","letra3","letra4", "letra5", "letra6"]]
    #concateno
    datos_bk = pd.concat([datos_bk.reset_index(drop = True), datos_bk_a_ciiu.reset_index(drop = True)], axis = 1)
    return datos_bk


def predo_ciiu_letra(dic_ciiu, comercio):

    ciiu3_4c_comercio = dic_ciiu[dic_ciiu["ciiu3_letra"]=="G"]["ciiu3_4c"]
    
    ciiu_letra =  dic_ciiu[["ciiu3_4c", "ciiu3_letra"]]
    ciiu_letra = ciiu_letra [~ciiu_letra ["ciiu3_4c"].isin(ciiu3_4c_comercio)]
    ciiu_letra  = pd.concat([ciiu_letra  , comercio[["clae6", "letra"]].rename(columns = {"clae6":"ciiu3_4c", "letra":"ciiu3_letra"})], axis = 0)
    ciiu_letra["ciiu3_4c"] = ciiu_letra["ciiu3_4c"].astype(str)
    return ciiu_letra
   
def dic_clae_and_ciiu(clae_to_ciiu,clae, dic_ciiu):
    ciiu3_4c_comercio = dic_ciiu[dic_ciiu["ciiu3_letra"]=="G"]["ciiu3_4c"] # vector de filtro

    # clae_to_ciiu.dropna(inplace = True)
    clae_to_ciiu["ciiu3_4c"] = clae_to_ciiu["ciiu3_4c"].astype(int).astype(str)
    
    clae_to_ciiu = clae_to_ciiu[clae_to_ciiu["clae6"] !=332000] 
    # clae_to_ciiu["ciiu3_4c"] = np.where(clae_to_ciiu["clae6"] ==332000, 
    #                                     "29_30_31_32_33", 
    #                                     clae_to_ciiu["ciiu3_4c"]  )
    # clae_to_ciiu.drop_duplicates(subset= "clae6", inplace = True) 

    claes_faltantes = pd.DataFrame({'clae6': [204000, 523032, 462110, 332000], 'clae6_desc': ["Servicios industriales para la fabricación de sustancias y productos quí­micos", 
                                                                         "Servicios de operadores logísticos seguros (OLS) en el Ámbito aduanero",
                                                                         "Acopio de algodón", "Instalación de maquinaria y equipos industriales" 
                                                       ] , 'ciiu3_4c': ["2429", "6350", "5121", "29_30_31_32_33"]})    
    clae_to_ciiu = pd.concat([clae_to_ciiu, claes_faltantes ], axis = 0)

    clae_to_ciiu_sin_g = clae_to_ciiu[~clae_to_ciiu["ciiu3_4c"].isin(ciiu3_4c_comercio )]
    clae_to_ciiu_sin_g.sort_values(by = "clae6", ascending = True, inplace = True)
    
    #3 
    clae["ciiu3_4c"] = clae["clae6"]
    clae["clae6"] = clae["clae6"].astype(int)
    clae_comercio = clae[clae["letra"] =="G"][["clae6","clae6_desc", "ciiu3_4c"]]
    
    clae_to_ciiu_mod = pd.concat([clae_to_ciiu_sin_g ,clae_comercio], 0)
    
    return clae_to_ciiu_mod


################################
# CLASIFICACION TIPO BIEN
################################

def filtro_stp(dic_stp, impo_bec):
    # vectores filtros
    # stp_general = dic_stp[dic_stp["utilizacion"]=="General"] # =~ CI
    stp_especifico = dic_stp[dic_stp["utilizacion"].str.contains("Específico|Transporte", case = False)] # =~ BK
    #join_impo_clae_bec_bk["dest_clean"] = join_impo_clae_bec_bk["destinacion"].apply(lambda x: destinacion_limpio(x))

    # opción 1
    impo_bec = impo_bec[impo_bec["BEC5EndUse"].str.startswith("CAP", na=False)]
    impo_bec["ue_dest"] = np.where(impo_bec["HS6"].isin(stp_especifico ["NCM"]), "BK", "")
    #impo_bec["ue_dest"].value_counts()

    # filtrado 1
    filtro1 = impo_bec[impo_bec["ue_dest"]==""]

    return  filtro1, impo_bec

def clasificacion_BK(filtro1):

    filtro1st =  filtro1[filtro1["dest_clean"] == "S/TR"]
    filtro1ct =  filtro1[filtro1["dest_clean"] == "C/TR"]
    filtro1co =  filtro1[filtro1["dest_clean"] == "CONS&Otros"]

    len(filtro1)  == ( len(filtro1st) +len(filtro1ct) + len(filtro1co) )

    # Filtros de conjuntos
    set_st = set(filtro1st["HS6_d12"])
    set_ct = set(filtro1ct["HS6_d12"])
    set_co = set(filtro1co["HS6_d12"])

    filtro_a = set_st - set_co - set_ct
    filtro_b = set_ct - set_st - set_co
    filtro_c = set_co - set_ct -set_st

    filtro_d = (set_st & set_co) - (set_ct )
    filtro_e = (set_ct & set_co) - ( set_st)
    filtro_f = (set_ct & set_st) - set_co
    filtro_g = set_ct & set_st & set_co

    dest_a = filtro1[filtro1["HS6_d12"].isin( filtro_a  )]
    dest_b = filtro1[filtro1["HS6_d12"].isin( filtro_b  )]
    dest_c = filtro1[filtro1["HS6_d12"].isin( filtro_c  )]
    dest_d = filtro1[filtro1["HS6_d12"].isin( filtro_d  )]
    dest_e = filtro1[filtro1["HS6_d12"].isin( filtro_e  )]
    dest_f = filtro1[filtro1["HS6_d12"].isin( filtro_f  )]
    dest_g = filtro1[filtro1["HS6_d12"].isin( filtro_g )]

    dest_a["filtro"] = "A"
    dest_b["filtro"] = "B"
    dest_c["filtro"] = "C"
    dest_d["filtro"] = "D"
    dest_e["filtro"] = "E"
    dest_f["filtro"] = "F"
    dest_g["filtro"] = "G"


    # Consistencia filtro 2
    #importaciones detectadas con S/T, C/T y Consumo
    len(filtro1st) +len(filtro1ct) + len(filtro1co)

    (len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) +len(dest_b)+ len(dest_c)) ==len(filtro1)

    # Concateno el filtro 2
    filtro1 = pd.concat( [dest_a, dest_b, dest_c, dest_d,  dest_e,  dest_f, dest_g], axis = 0)


    # filtro1["ue_dest"].value_counts()
    filtro1["ue_dest"] = np.where((filtro1["filtro"] == "A") &
                                    (filtro1["dest_clean"] == "S/TR"),
                                        "BK",
                                        np.where( (filtro1["filtro"] == "B") &
                                                (filtro1["dest_clean"] == "C/TR"),
                                                "CI", ""
                                                )
                                    )

    filtro1[["ue_dest", "filtro"]].value_counts()


    filtro2 = filtro1[filtro1["ue_dest"] == "" ]
    # len(filtro2)/len(filtro1)-1

    # filtro2["filtro"].value_counts()
    # filtro2[["dest_clean", "filtro"]].value_counts()
    # filtro2[["dest_clean","filtro", "uni_decl" ]].value_counts()
    # filtro2[["dest_clean","filtro", "uni_est" ]].value_counts()



    # =============================================================================
    # ## Aplicando métrica
    # =============================================================================

    # Aplico métrica 1
    # dest_c["metric"] = dest_c.apply(lambda x : (x["valor"] * x["kilos"])/x["cant_decl"] , axis = 1)
    filtro2["metric"] = filtro2.apply(lambda x : metrica(x), axis = 1)

    # CASO D
    #con uni decl
    # filtro2_centerD = filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
    # filtro2_D = pd.merge(filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_centerD.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

    #con uni est
    filtro2_D_mad= filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
    filtro2_D_median = filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
    filtro2_D = pd.merge(filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_D_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])
    filtro2_D = pd.merge(filtro2_D, filtro2_D_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

    filtro2_D["mad"] = np.where(filtro2_D["mad"]==0, 0.001, filtro2_D["mad"]  )
    filtro2_D["brecha"] = (filtro2_D["metric"]/filtro2_D["median"])-1
    filtro2_D["z_score"] = filtro2_D.apply(lambda x: 0.6745*((x["metric"]-x["median"]))/(x["mad"]), axis =1 )
    filtro2_D["ue_dest"] = np.where(filtro2_D["z_score"] <= -3.5, "CI","BK" ) # no conviene establecer los límites al reves? con un -3.5??

    filtro2_D_bk = filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ]
    filtro2_D_bk["ue_dest"] = "BK"

    clasif_D = pd.concat([filtro2_D_bk, filtro2_D.drop(["median", "mad", "brecha", "z_score"], axis = 1)] )
    clasif_D["ue_dest"].value_counts()
    clasif_D.groupby(["ue_dest"])["valor"].sum()

    # CASO E
    #con uni est
    filtro2_E_mad= filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
    filtro2_E_median = filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
    filtro2_E = pd.merge(filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_E_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])
    filtro2_E = pd.merge(filtro2_E, filtro2_E_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

    filtro2_E["mad"] = np.where(filtro2_E["mad"]==0, 0.001, filtro2_E["mad"]  )
    filtro2_E["brecha"] = (filtro2_E["metric"]/filtro2_E["median"])-1
    filtro2_E["z_score"] = filtro2_E.apply(lambda x: 0.6745*((x["metric"]-x["median"]))/(x["mad"]), axis =1 )
    filtro2_E["ue_dest"] = np.where(filtro2_E["z_score"] <= 3.5, "CI","BK" )

    filtro2_E_ci = filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="C/TR") ]
    filtro2_E_ci["ue_dest"] = "CI"

    clasif_E = pd.concat([filtro2_E_ci, filtro2_E.drop(["median", "mad", "brecha", "z_score"], axis = 1) ])
    clasif_E["ue_dest"].value_counts()


    # CASO G
    # filtro2[filtro2["filtro"]=="G"].value_counts("dest_clean")
    filtro2_G_mad_bk = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12" , "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_bk"}).reset_index(drop = True)
    filtro2_G_median_bk = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_bk"}).reset_index(drop = True)
    filtro2_G_mad_ci = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_ci"}).reset_index(drop = True)
    filtro2_G_median_ci = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_ci"}).reset_index(drop = True)

    filtro2_G = pd.merge(filtro2[(filtro2["filtro"]=="G")  & (filtro2["dest_clean"]=="CONS&Otros")] , filtro2_G_mad_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left", left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
    filtro2_G = pd.merge(filtro2_G , filtro2_G_median_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
    filtro2_G = pd.merge(filtro2_G , filtro2_G_mad_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
    filtro2_G = pd.merge(filtro2_G , filtro2_G_median_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])

    filtro2_G ["mad_bk"] = np.where(filtro2_G ["mad_bk"]==0, 0.001, filtro2_G ["mad_bk"]  )
    filtro2_G ["mad_ci"] = np.where(filtro2_G ["mad_ci"]==0, 0.001, filtro2_G ["mad_ci"]  )
    # filtro2_G ["brecha"] = (filtro2_G ["metric"]/filtro2_G ["median"])-1
    filtro2_G ["z_score_bk"] = filtro2_G .apply(lambda x: 0.6745*((x["metric"]-x["median_bk"]))/(x["mad_bk"]), axis =1 )
    filtro2_G ["z_score_ci"] = filtro2_G .apply(lambda x: 0.6745*((x["metric"]-x["median_ci"]))/(x["mad_ci"]), axis =1 )

    filtro2_G["ue_dest"] = np.where(np.abs(filtro2_G["z_score_ci"]) > np.abs(filtro2_G["z_score_bk"]), "BK", "CI" )
    filtro2_G["ue_dest"].value_counts()

    filtro2_G_clas = filtro2[(filtro2["filtro"]=="G")  & (filtro2["dest_clean"]!="CONS&Otros")]
    filtro2_G_clas ["ue_dest"] = np.where(filtro2_G_clas["dest_clean"]== "S/TR","BK", "CI" )
    filtro2_G_clas["ue_dest"].value_counts()

    clasif_G = pd.concat([filtro2_G_clas, filtro2_G.drop(["median_bk", "median_ci",  "mad_bk", "mad_ci", "z_score_bk", "z_score_ci"],axis =1 )], axis = 0)
    clasif_G ["ue_dest"].value_counts()


    ## EDA de Datos ya clasificados
    clasif_AB = filtro1[filtro1["ue_dest"] != "" ]
    clasif_AB["metric"] = clasif_AB.apply(lambda x: metrica(x), axis = 1)

    data_clasif = pd.concat([clasif_AB, clasif_D ,clasif_E,clasif_G  ], axis= 0)
    # data_clasif["metric_zscore"] = (data_clasif["metric"] -data_clasif["metric"].mean())/ data_clasif["metric"].std(ddof=1)

    data_clasif[["filtro", "ue_dest"]].value_counts()

    ###############################################
    ## DATOS NO CLASIFICADOS
    data_not_clasif = pd.concat([dest_c], axis=0)
    data_not_clasif["ue_dest"] = "?"  # np.NAN

    # data_not_clasif.isna().sum()
    data_not_clasif["metric"] = data_not_clasif.apply(lambda x: metrica(x), axis=1)
    data_not_clasif["precio_kilo"] = data_not_clasif["valor"] / data_not_clasif["kilos"]

    # data_not_clasif.to_csv("../data/resultados/bk_sin_ue_dest.csv")


    return data_clasif, data_not_clasif

# JOIN CON DATOS STP Y CLASIFICADOS PROPIOS
def join_stp_clasif_prop(join_impo_clae_bec_bk, data_clasif):
    stp_ue_dest = join_impo_clae_bec_bk[join_impo_clae_bec_bk["ue_dest"]=="BK" ]
    stp_ue_dest ["ue_dest"].value_counts()
    stp_ue_dest ["metric"] = stp_ue_dest .apply(lambda x: metrica(x), axis = 1)
    stp_ue_dest["filtro"] = "STP"

    data_clasif_ue_dest = pd.concat([data_clasif, stp_ue_dest], axis = 0)
    data_clasif_ue_dest ["ue_dest"].value_counts()

    data_clasif_ue_dest["precio_kilo"]= data_clasif_ue_dest["valor"]/data_clasif_ue_dest["kilos"]

    return  data_clasif_ue_dest


def clasificacion_CI(impo_bec):
    impo_bec_ci = impo_bec[impo_bec["BEC5EndUse"].str.startswith("INT", na=False)]
    impo_bec_ci["dest_clean"].value_counts()  # .sum()

    filtro1st = impo_bec_ci[impo_bec_ci["dest_clean"] == "S/TR"]
    filtro1ct = impo_bec_ci[impo_bec_ci["dest_clean"] == "C/TR"]
    filtro1co = impo_bec_ci[impo_bec_ci["dest_clean"] == "CONS&Otros"]

    len(impo_bec_ci) == (len(filtro1st) + len(filtro1ct) + len(filtro1co))

    # Filtros de conjuntos
    set_st = set(filtro1st["HS6_d12"])
    set_ct = set(filtro1ct["HS6_d12"])
    set_co = set(filtro1co["HS6_d12"])

    filtro_a = set_st - set_co - set_ct
    filtro_b = set_ct - set_st - set_co
    filtro_c = set_co - set_ct - set_st

    filtro_d = (set_st & set_co) - (set_ct)
    filtro_e = (set_ct & set_co) - (set_st)
    filtro_f = (set_ct & set_st) - set_co
    filtro_g = set_ct & set_st & set_co

    dest_a = impo_bec_ci[impo_bec_ci["HS6_d12"].isin(filtro_a)]
    dest_b = impo_bec_ci[impo_bec_ci["HS6_d12"].isin(filtro_b)]
    dest_c = impo_bec_ci[impo_bec_ci["HS6_d12"].isin(filtro_c)]
    dest_d = impo_bec_ci[impo_bec_ci["HS6_d12"].isin(filtro_d)]
    dest_e = impo_bec_ci[impo_bec_ci["HS6_d12"].isin(filtro_e)]
    dest_f = impo_bec_ci[impo_bec_ci["HS6_d12"].isin(filtro_f)]
    dest_g = impo_bec_ci[impo_bec_ci["HS6_d12"].isin(filtro_g)]

    (len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) + len(dest_b) + len(dest_c)) == len(
        impo_bec_ci)

    dest_a["filtro"] = "A"
    dest_b["filtro"] = "B"
    dest_c["filtro"] = "C"
    dest_d["filtro"] = "D"
    dest_e["filtro"] = "E"
    dest_f["filtro"] = "F"
    dest_g["filtro"] = "G"

    cons_int = pd.concat([dest_a, dest_b, dest_c, dest_d, dest_e, dest_f, dest_g], axis=0)
    cons_int["metric"] = cons_int.apply(lambda x: metrica(x), axis=1)

    # A , B, C, E
    cons_int["ue_dest"] = np.where((cons_int["filtro"] == "B") |
                                   (cons_int["filtro"] == "E") |
                                   (cons_int["filtro"] == "C"), "CI",
                                   np.where((cons_int["filtro"] == "A"), "BK",
                                            np.nan))
    # D
    cons_int_D_mad = cons_int[(cons_int["filtro"] == "D") & (cons_int["dest_clean"] == "S/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(
        lambda x: median_abs_deviation(x)).rename(columns={"metric": "mad"}).reset_index(drop=True)
    cons_int_D_median = cons_int[(cons_int["filtro"] == "D") & (cons_int["dest_clean"] == "S/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(lambda x: x.median()).rename(
        columns={"metric": "median"}).reset_index(drop=True)
    cons_int_D = pd.merge(cons_int[(cons_int["filtro"] == "D") & (cons_int["dest_clean"] == "CONS&Otros")],
                          cons_int_D_mad.drop("dest_clean", axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl", "filtro"], right_on=["HS6_d12", "uni_decl", "filtro"])
    cons_int_D = pd.merge(cons_int_D, cons_int_D_median.drop("dest_clean", axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl", "filtro"], right_on=["HS6_d12", "uni_decl", "filtro"])

    cons_int_D["mad"] = np.where(cons_int_D["mad"] == 0, 0.001, cons_int_D["mad"])
    cons_int_D["z_score"] = cons_int_D.apply(lambda x: 0.6745 * ((x["metric"] - x["median"])) / (x["mad"]), axis=1)
    cons_int_D["ue_dest"] = np.where(cons_int_D["z_score"] > 3.5, "BK", "CI")

    cons_int_D_st = cons_int[(cons_int["filtro"] == "D") & (cons_int["dest_clean"] == "S/TR")]
    cons_int_D_st["ue_dest"] = "BK"

    cons_int_D = pd.concat([cons_int_D, cons_int_D_st], axis=0)
    # cons_int_D["ue_dest"].value_counts()

    # CASO G
    # cons_int[cons_int["filtro"]=="G"].value_counts("dest_clean")
    cons_int_G_mad_bk = cons_int[(cons_int["filtro"] == "G") & (cons_int["dest_clean"] == "S/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(
        lambda x: median_abs_deviation(x)).rename(columns={"metric": "mad_bk"}).reset_index(drop=True)
    cons_int_G_median_bk = cons_int[(cons_int["filtro"] == "G") & (cons_int["dest_clean"] == "S/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(lambda x: x.median()).rename(
        columns={"metric": "median_bk"}).reset_index(drop=True)
    cons_int_G_mad_ci = cons_int[(cons_int["filtro"] == "G") & (cons_int["dest_clean"] == "C/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(
        lambda x: median_abs_deviation(x)).rename(columns={"metric": "mad_ci"}).reset_index(drop=True)
    cons_int_G_median_ci = cons_int[(cons_int["filtro"] == "G") & (cons_int["dest_clean"] == "C/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(lambda x: x.median()).rename(
        columns={"metric": "median_ci"}).reset_index(drop=True)

    cons_int_G = pd.merge(cons_int[(cons_int["filtro"] == "G") & (cons_int["dest_clean"] == "CONS&Otros")],
                          cons_int_G_mad_bk.drop(["dest_clean", "filtro"], axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl"], right_on=["HS6_d12", "uni_decl"])
    cons_int_G = pd.merge(cons_int_G, cons_int_G_median_bk.drop(["dest_clean", "filtro"], axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl"], right_on=["HS6_d12", "uni_decl"])
    cons_int_G = pd.merge(cons_int_G, cons_int_G_mad_ci.drop(["dest_clean", "filtro"], axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl"], right_on=["HS6_d12", "uni_decl"])
    cons_int_G = pd.merge(cons_int_G, cons_int_G_median_ci.drop(["dest_clean", "filtro"], axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl"], right_on=["HS6_d12", "uni_decl"])

    cons_int_G["mad_bk"] = np.where(cons_int_G["mad_bk"] == 0, 0.001, cons_int_G["mad_bk"])
    cons_int_G["mad_ci"] = np.where(cons_int_G["mad_ci"] == 0, 0.001, cons_int_G["mad_ci"])
    cons_int_G["z_score_bk"] = cons_int_G.apply(lambda x: 0.6745 * ((x["metric"] - x["median_bk"])) / (x["mad_bk"]),
                                                axis=1)
    cons_int_G["z_score_ci"] = cons_int_G.apply(lambda x: 0.6745 * ((x["metric"] - x["median_ci"])) / (x["mad_ci"]),
                                                axis=1)

    cons_int_G["ue_dest"] = np.where(np.abs(cons_int_G["z_score_ci"]) > np.abs(cons_int_G["z_score_bk"]), "BK", "CI")
    cons_int_G["ue_dest"].value_counts()

    cons_int_G_no_cons = cons_int[(cons_int["filtro"] == "G") & (cons_int["dest_clean"] != "CONS&Otros")]
    cons_int_G_no_cons["ue_dest"] = np.where(cons_int_G_no_cons["dest_clean"] == "S/TR", "BK", "CI")
    # cons_int_G_clas["ue_dest"].value_counts()

    cons_int_G_clasif = pd.concat([cons_int_G_no_cons, cons_int_G.drop(
        ["median_bk", "median_ci", "mad_bk", "mad_ci", "z_score_bk", "z_score_ci"], axis=1)], axis=0)
    cons_int_G_clasif["ue_dest"].value_counts()  # .sum()

    cons_int_clasif = pd.concat([cons_int_D, cons_int_G_clasif, cons_int[cons_int["filtro"].str.contains("A|B|C|E")]])

    return cons_int_clasif ,impo_bec_ci


def clasificacion_CONS(impo_bec):
    impo_bec_cons = impo_bec[impo_bec["BEC5EndUse"].str.startswith("CONS", na=False)]
    impo_bec_cons["dest_clean"].value_counts()  # .sum()

    filtro1st = impo_bec_cons[impo_bec_cons["dest_clean"] == "S/TR"]
    filtro1ct = impo_bec_cons[impo_bec_cons["dest_clean"] == "C/TR"]
    filtro1co = impo_bec_cons[impo_bec_cons["dest_clean"] == "CONS&Otros"]

    len(impo_bec_cons) == (len(filtro1st) + len(filtro1ct) + len(filtro1co))

    # Filtros de conjuntos
    set_st = set(filtro1st["HS6_d12"])
    set_ct = set(filtro1ct["HS6_d12"])
    set_co = set(filtro1co["HS6_d12"])

    filtro_a = set_st - set_co - set_ct
    filtro_b = set_ct - set_st - set_co
    filtro_c = set_co - set_ct - set_st

    filtro_d = (set_st & set_co) - (set_ct)
    filtro_e = (set_ct & set_co) - (set_st)
    filtro_f = (set_ct & set_st) - set_co
    filtro_g = set_ct & set_st & set_co

    dest_a = impo_bec_cons[impo_bec_cons["HS6_d12"].isin(filtro_a)]
    dest_b = impo_bec_cons[impo_bec_cons["HS6_d12"].isin(filtro_b)]
    dest_c = impo_bec_cons[impo_bec_cons["HS6_d12"].isin(filtro_c)]
    dest_d = impo_bec_cons[impo_bec_cons["HS6_d12"].isin(filtro_d)]
    dest_e = impo_bec_cons[impo_bec_cons["HS6_d12"].isin(filtro_e)]
    dest_f = impo_bec_cons[impo_bec_cons["HS6_d12"].isin(filtro_f)]
    dest_g = impo_bec_cons[impo_bec_cons["HS6_d12"].isin(filtro_g)]

    (len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) + len(dest_b) + len(dest_c)) == len(
        impo_bec_cons)

    dest_a["filtro"] = "A"
    dest_b["filtro"] = "B"
    dest_c["filtro"] = "C"
    dest_d["filtro"] = "D"
    dest_e["filtro"] = "E"
    dest_f["filtro"] = "F"
    dest_g["filtro"] = "G"

    cons_fin = pd.concat([dest_a, dest_b, dest_c, dest_d, dest_e, dest_f, dest_g], axis=0)

    # A, B, C, E
    cons_fin["ue_dest"] = np.where((cons_fin["filtro"] == "B") |
                                   (cons_fin["filtro"] == "E") |
                                   (cons_fin["filtro"] == "C"), "CI",
                                   np.where((cons_fin["filtro"] == "A"), "BK",
                                            np.nan))
    cons_fin["ue_dest"].value_counts()

    cons_fin["metric"] = cons_fin.apply(lambda x: metrica(x), axis=1)

    # D
    # con uni est
    cons_fin_D_mad = cons_fin[(cons_fin["filtro"] == "D") & (cons_fin["dest_clean"] == "S/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(
        lambda x: median_abs_deviation(x)).rename(columns={"metric": "mad"}).reset_index(drop=True)
    cons_fin_D_median = cons_fin[(cons_fin["filtro"] == "D") & (cons_fin["dest_clean"] == "S/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(lambda x: x.median()).rename(
        columns={"metric": "median"}).reset_index(drop=True)
    cons_fin_D = pd.merge(cons_fin[(cons_fin["filtro"] == "D") & (cons_fin["dest_clean"] == "CONS&Otros")],
                          cons_fin_D_mad.drop("dest_clean", axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl", "filtro"], right_on=["HS6_d12", "uni_decl", "filtro"])
    cons_fin_D = pd.merge(cons_fin_D, cons_fin_D_median.drop("dest_clean", axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl", "filtro"], right_on=["HS6_d12", "uni_decl", "filtro"])

    cons_fin_D["mad"] = np.where(cons_fin_D["mad"] == 0, 0.001, cons_fin_D["mad"])
    cons_fin_D["brecha"] = (cons_fin_D["metric"] / cons_fin_D["median"]) - 1
    cons_fin_D["z_score"] = cons_fin_D.apply(lambda x: 0.6745 * ((x["metric"] - x["median"])) / (x["mad"]), axis=1)
    cons_fin_D["ue_dest"] = np.where(cons_fin_D["z_score"] > 3.5, "BK", "CI")

    cons_fin_D_st = cons_fin[(cons_fin["filtro"] == "D") & (cons_fin["dest_clean"] == "S/TR")]
    cons_fin_D_st["ue_dest"] = "BK"

    cons_fin_D = pd.concat([cons_fin_D, cons_fin_D_st], axis=0)
    # cons_fin_D["ue_dest"].value_counts()#.sum()

    # CASO G
    # cons_fin[cons_fin["filtro"]=="G"].value_counts("dest_clean")
    cons_fin_G_mad_bk = cons_fin[(cons_fin["filtro"] == "G") & (cons_fin["dest_clean"] == "S/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(
        lambda x: median_abs_deviation(x)).rename(columns={"metric": "mad_bk"}).reset_index(drop=True)
    cons_fin_G_median_bk = cons_fin[(cons_fin["filtro"] == "G") & (cons_fin["dest_clean"] == "S/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(lambda x: x.median()).rename(
        columns={"metric": "median_bk"}).reset_index(drop=True)
    cons_fin_G_mad_ci = cons_fin[(cons_fin["filtro"] == "G") & (cons_fin["dest_clean"] == "C/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(
        lambda x: median_abs_deviation(x)).rename(columns={"metric": "mad_ci"}).reset_index(drop=True)
    cons_fin_G_median_ci = cons_fin[(cons_fin["filtro"] == "G") & (cons_fin["dest_clean"] == "C/TR")].groupby(
        ["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index=False)["metric"].apply(lambda x: x.median()).rename(
        columns={"metric": "median_ci"}).reset_index(drop=True)

    cons_fin_G = pd.merge(cons_fin[(cons_fin["filtro"] == "G") & (cons_fin["dest_clean"] == "CONS&Otros")],
                          cons_fin_G_mad_bk.drop(["dest_clean", "filtro"], axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl"], right_on=["HS6_d12", "uni_decl"])
    cons_fin_G = pd.merge(cons_fin_G, cons_fin_G_median_bk.drop(["dest_clean", "filtro"], axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl"], right_on=["HS6_d12", "uni_decl"])
    cons_fin_G = pd.merge(cons_fin_G, cons_fin_G_mad_ci.drop(["dest_clean", "filtro"], axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl"], right_on=["HS6_d12", "uni_decl"])
    cons_fin_G = pd.merge(cons_fin_G, cons_fin_G_median_ci.drop(["dest_clean", "filtro"], axis=1), how="left",
                          left_on=["HS6_d12", "uni_decl"], right_on=["HS6_d12", "uni_decl"])

    cons_fin_G["mad_bk"] = np.where(cons_fin_G["mad_bk"] == 0, 0.001, cons_fin_G["mad_bk"])
    cons_fin_G["mad_ci"] = np.where(cons_fin_G["mad_ci"] == 0, 0.001, cons_fin_G["mad_ci"])
    cons_fin_G["z_score_bk"] = cons_fin_G.apply(lambda x: 0.6745 * ((x["metric"] - x["median_bk"])) / (x["mad_bk"]),
                                                axis=1)
    cons_fin_G["z_score_ci"] = cons_fin_G.apply(lambda x: 0.6745 * ((x["metric"] - x["median_ci"])) / (x["mad_ci"]),
                                                axis=1)

    cons_fin_G["ue_dest"] = np.where(np.abs(cons_fin_G["z_score_ci"]) > np.abs(cons_fin_G["z_score_bk"]), "BK", "CI")
    # cons_fin_G["ue_dest"].value_counts()

    cons_fin_G_no_cons = cons_fin[(cons_fin["filtro"] == "G") & (cons_fin["dest_clean"] != "CONS&Otros")]
    cons_fin_G_no_cons["ue_dest"] = np.where(cons_fin_G_no_cons["dest_clean"] == "S/TR", "BK", "CI")
    # cons_fin_G_no_cons["ue_dest"].value_counts()

    cons_fin_G_clasif = pd.concat([cons_fin_G_no_cons, cons_fin_G.drop(
        ["median_bk", "median_ci", "mad_bk", "mad_ci", "z_score_bk", "z_score_ci"], axis=1)], axis=0)
    # cons_fin_G_clasif  ["ue_dest"].value_counts()

    cons_fin_clasif = pd.concat([cons_fin_D, cons_fin_G_clasif, cons_fin[cons_fin["filtro"].str.contains("A|B|C|E")]])
    return cons_fin_clasif, impo_bec_cons

def datos_modelo(cons_fin_clasif, cons_int_clasif,data_clasif_ue_dest,data_not_clasif):
    # impo_ue_dest = pd.concat([pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha", 'metric', 'ue_dest', 'mad', 'median', 'z_score'], axis = 1), bk], axis =0)
    cicf_ue_dest = pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha",  'mad', 'median', 'z_score'], axis = 1) #, bk], axis =0)
    cicf_ue_dest["precio_kilo"] =  cicf_ue_dest["valor"]/cicf_ue_dest["kilos"]

    # bk_ue_dest = pd.read_csv("../data/resultados/bk_con_ue_dest.csv")
    bk_ue_dest = data_clasif_ue_dest#.copy().drop(['HS4', 'HS4Desc', 'HS6Desc', "BEC5Category"], 1)

    # bk_sin_ue_dest = pd.read_csv("../data/resultados/bk_sin_ue_dest.csv")
    bk_sin_ue_dest = data_not_clasif#.drop(['HS4', 'HS4Desc', 'HS6Desc', "BEC5Category"], 1)

    print((len(join_impo_clae)-  len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )) - ( len(bk_sin_ue_dest ) + len(bk_ue_dest)+ len(cicf_ue_dest) ))

    data_model = pd.concat([bk_sin_ue_dest , bk_ue_dest, cicf_ue_dest ], axis = 0)


    #PREPROCESAMIENTO
    data_model ['HS6'] = data_model ['HS6'].astype("str")
    data_model ['HS8'] = data_model ['HS6_d12'].str.slice(0,8)
    data_model ['HS10'] = data_model ['HS6_d12'].str.slice(0,10)

    data_model["actividades"] = data_model["letra1"] + data_model["letra2"] + data_model["letra3"] + data_model[
        "letra4"] + data_model["letra5"] + data_model["letra6"]
    data_model["act_ordenadas"] = data_model["actividades"].apply(
        lambda x: "".join(sorted(x)))  # "".join(sorted(data_model["actividades"]))


    # preprocesamiento etiquetados
    cols = ["HS6", "HS8", "HS10",
            'valor', 'kilos', "precio_kilo",
            "letra1", "letra2", "letra3", "letra4", "letra5", "letra6",
            "act_ordenadas",
            "uni_est", "cant_est", "uni_decl", "cant_decl",
            "metric",
            "ue_dest"]

    data_model = data_model[cols]

    data_model.to_csv("../data/resultados/data_modelo_diaria.csv", index=False)

    return data_model

