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
import datetime


# Script 1
def predo_ncm12_desc(ncm12_desc ):
    ncm12_desc = ncm12_desc[["POSICION", "DESCRIPCIO"]]
    ncm12_desc.rename(columns = {"POSICION": "HS_12d", "DESCRIPCIO":"hs6_d12_desc"}, inplace = True) 
    ncm12_desc_split = pd.concat([ncm12_desc.iloc[:,0], pd.DataFrame(ncm12_desc['hs6_d12_desc'].str.split('//', expand=True))], axis=1)
    dic = {"ncm_desc": ncm12_desc, "ncm_split": ncm12_desc_split }
    
    dic["ncm_desc"].to_csv("../data/resultados/ncm_12digits.csv", index =False) 
    return dic["ncm_desc"]


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

# def predo_impo_all(impo_d12, name_file):
#     impo_d12["anyo"] = impo_d12["FECH_OFIC"].str.slice(0,4)
#
#     #impo_d12 = impo_d12[['anyo','CUIT_IMPOR', 'NOMBRE']]
#     impo_d12 = impo_d12[impo_d12["anyo"]=="2017"]
#     #impo_d12 = impo_d12.drop_duplicates()
#
#     impo_d12.to_csv("../data/"+name_file, index = False)
#     return impo_d12


def predo_impo_17(impo_17):
    impo_17['FOB'] = impo_17['FOB'].str.replace(",", ".")
    impo_17['FOB'] = impo_17['FOB'].astype(float)
    impo_17['CIF'] = impo_17['CIF'].str.replace(",", ".")
    impo_17['CIF'] = impo_17['CIF'].astype(float)
    impo_17.drop("FOB", axis=1, inplace=True)
    return impo_17.rename(columns = {"POSIC_SIM" : "HS6", 'CIF': "valor"} , inplace = True)


def predo_impo_12d(impo_d12, ncm_desc):
    impo_d12["ANYO"] = impo_d12["FECH_OFIC"].str.slice(0,4)
    impo_d12 = impo_d12[impo_d12["ANYO"] == "2017"]             # FILTRO AÑO 2017

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
    for clae_i in ["actividad2", "actividad3", "actividad4", "actividad5", "actividad6"]:
        cuit_clae6.loc[cuit_clae6['suma']<1, [clae_i]] = cuit_clae6['actividad1']
    
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
    #clae["clae6"] = clae["clae6"].astype("Int64").astype(str)
    clae["clae6"] = clae["clae6"].astype(str)
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
    
     
def predo_stp(dic_stp ):
    dic_stp.columns = ["NCM", "desc", "ciiu", "desc_gral", "utilizacion", "demanda"]
    dic_stp.dropna(thresh = 3, inplace= True)
    # agro_stp = dic_stp[dic_stp["demanda"].str.contains("agrí", case = False)]

    return dic_stp

def predo_ncm(ncm_sector):
    ncm_sector["hs6"] = ncm_sector["hs6"].astype(int)
    return ncm_sector


def predo_dic_propio(clae_to_ciiu, dic_ciiu,clae):
    # Join ciiu digitos con letra (posee clae para hacer el join)
    clae_to_ciiu.loc[clae_to_ciiu["ciiu3_4c"].isnull(),"ciiu3_4c"  ] = 4539
    ciiu_dig_let = pd.merge(dic_ciiu[["ciiu3_4c", "ciiu3_letra"]], clae_to_ciiu.drop("clae6_desc", 1), left_on = "ciiu3_4c", right_on = "ciiu3_4c" , how = "inner") #"ciiu3_4c_desc"
    
    #convierto a string los digitos del ciiu
    ciiu_dig_let["ciiu3_4c"] = ciiu_dig_let["ciiu3_4c"].astype(int).astype(str)
    ciiu_dig_let["clae6"] = ciiu_dig_let["clae6"].astype(int).astype(str)
    clae["clae6"] = clae["clae6"].astype(int).astype(str)
    
    #filtro el caso problematico del clae    
    ciiu_dig_let = ciiu_dig_let[ciiu_dig_let["clae6"] !="332000"] 
    
    #agrego los clae faltantes
    claes_faltantes = pd.DataFrame({'clae6': ["204000", "523032", "462110", "332000", np.nan], 'ciiu3_letra': ["D", "I", "G", "D" ,"CONS" ] , 
                                    # "ciiu3_4c_desc" : ["", "", "", ""],
                                    'ciiu3_4c': ["2429", "6350", "5121", "29_30_31_32_33", "CONS"]})    
    ciiu_dig_let = pd.concat([ciiu_dig_let, claes_faltantes ], axis = 0)
  
    ciiu_dig_let = pd.merge(ciiu_dig_let , clae[["clae6", "letra"]], how = "left", left_on= "clae6", right_on= "clae6").rename(columns = {"letra": "clae6_letra"})
    
    # ciiu_dig_let["clae6"] = ciiu_dig_let["clae6"].astype(int).astype(str)
    ciiu_dig_let["propio"] = np.where((ciiu_dig_let["clae6_letra"]=="G") |(ciiu_dig_let["clae6"]=="99000") ,ciiu_dig_let["clae6"], ciiu_dig_let["ciiu3_4c"]   )#.astype(str)  
    ciiu_dig_let["propio_letra"] = np.where(ciiu_dig_let["clae6_letra"]=="G", ciiu_dig_let["clae6_letra"], ciiu_dig_let["ciiu3_letra"]   )
    ciiu_dig_let["propio_letra_2"] =np.where(ciiu_dig_let["propio_letra"].isin( ["I"]),   ciiu_dig_let["propio_letra"] +"_"+ ciiu_dig_let["propio"].str.slice(start=0,stop=2),
                                             np.where(ciiu_dig_let["propio_letra"].isin( ["H"]),   ciiu_dig_let["propio_letra"] +"_"+ ciiu_dig_let["propio"].str.slice(start=0,stop=3),
                                                      np.where(ciiu_dig_let["propio_letra"].isin( ["C"]),   ciiu_dig_let["propio_letra"] +"_"+ ciiu_dig_let["propio"].str.slice(start=0,stop=2),
                                                          np.where(ciiu_dig_let["clae6"]=="99000", ciiu_dig_let["propio_letra"] +"_"+ ciiu_dig_let["clae6"].str.slice(start=0,stop=2),
                                                                   np.where(ciiu_dig_let["propio_letra"]=="K", ciiu_dig_let["propio_letra"] +"_"+ ciiu_dig_let["propio"].str.slice(start=0,stop=2),
                                                                            np.where(ciiu_dig_let["propio_letra"]=="D", np.where(ciiu_dig_let["propio"]=="29_30_31_32_33", "D_29_30_31_32_33",
                                                                                                                                 "D"+"_"+ciiu_dig_let["propio"].str.slice(start=0,stop=2) 
                                                                                                                                 ), 
                                                                                     ciiu_dig_let["propio_letra"]
                                                                                     ) 
                                                                            )
                                                                   )
                                                          )
                                                 )
                                            )
    
                   
    #join descripcion                                   
    # desc = pd.read_csv("../data/resultados/desc_letra_propio_2.csv")#.drop("Unnamed: 2",1)
    desc = pd.read_csv("../data/resultados/desc_letra_propio_2.txt", sep = "\t", encoding="latin").drop("Unnamed: 2",1)
    desc.columns = ["letra", "desc"]
    ciiu_dig_let = pd.merge(ciiu_dig_let, desc, how= "left", left_on = "propio_letra_2", right_on = "letra")
    ciiu_dig_let.to_csv("../data/resultados/dic_clae_ciiu_propio.csv", index=False)

    return ciiu_dig_let


def diccionario_especial(datos, dic_propio):
    # conversion CLAE a CIIU (codigo para funcion)
    datos_bk_a_ciiu= datos
    
    ciiu_dig_let = dic_propio[["clae6", "propio", "propio_letra_2"]]
    # ESTE ES EL POSTA
    for clae_i, letra_i in zip(["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6"],
                               ["letra1", "letra2", "letra3", "letra4", "letra5", "letra6"]):
        
        # ciiu_data.rename(columns = {"ciiu3_4c": ciiu_name })
        datos_bk_a_ciiu[clae_i] = datos_bk_a_ciiu[clae_i].astype(int).astype(str)
        datos_bk_a_ciiu= pd.merge(datos_bk_a_ciiu , ciiu_dig_let, how = "left" ,left_on = clae_i, right_on= "clae6" ).drop(["clae6",clae_i,letra_i ], 1)
        datos_bk_a_ciiu.rename(columns = {"propio": clae_i, "propio_letra_2": letra_i  }, inplace = True)
    return datos_bk_a_ciiu


def def_join_impo_clae(impo_anyo_12d, cuit_empresas):
    impo_clae = pd.merge(impo_anyo_12d, cuit_empresas, left_on = "cuit", right_on = "CUIT", how = "right")
    impo_clae["dest_clean"] = impo_clae["destinacion"].apply(lambda x: destinacion_limpio(x))
    impo_clae.drop(["CUIT", "destinacion"], axis=1, inplace=True)

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


def def_join_comercio(join_impo_clae_bec_bk, comercio, ci = False):
    
    # join_impo_clae_bec_bk, comercio = datos_bk_sin_picks , vector_comercio_bk
    
    # conversion de vector de comercio para que se pueda realizar el merge
    comercio = comercio.copy()
    comercio["clae6"] = comercio["clae6"].astype(str)
    comercio.drop(["letra", "clae6_desc"] , axis = 1, inplace = True)
    
    if ci == False:    
        #comercio = comercio.drop(["vta_vehiculos"], axis=1)
        comercio2 = comercio.rename(columns = {"vta_bk": "vta_bk2", "vta_sec": "vta_sec2"})
        comercio3 = comercio.rename(columns = {"vta_bk": "vta_bk3", "vta_sec": "vta_sec3"})    
        comercio4 = comercio.rename(columns = {"vta_bk": "vta_bk4", "vta_sec": "vta_sec4"})
        comercio5 = comercio.rename(columns = {"vta_bk": "vta_bk5", "vta_sec": "vta_sec5"})
        comercio6 = comercio.rename(columns = {"vta_bk": "vta_bk6", "vta_sec": "vta_sec6"})
    elif ci == True:
        comercio2 = comercio.rename(columns = {"vta_sec": "vta_sec2"})
        comercio3 = comercio.rename(columns = {"vta_sec": "vta_sec3"})    
        comercio4 = comercio.rename(columns = {"vta_sec": "vta_sec4"})
        comercio5 = comercio.rename(columns = {"vta_sec": "vta_sec5"})
        comercio6 = comercio.rename(columns = {"vta_sec": "vta_sec6"})
            
    # join de la matriz con el sector comercio
    ## Comercio 1
    impo17_bec_complete = pd.merge(join_impo_clae_bec_bk, comercio, how = "left", left_on = "actividad1", right_on = "clae6").drop("clae6", axis=1) 
    ## Comercio 2
    impo17_bec_complete = pd.merge(impo17_bec_complete, comercio2, 
                                   how = "left", left_on = "actividad2", right_on = "clae6").drop("clae6", axis=1) 
    ## Comercio 3
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio3 , 
                          how = "left", left_on = "actividad3", right_on = "clae6").drop("clae6", axis=1) 
    ## Comercio 4
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio4 , 
                          how = "left", left_on = "actividad4", right_on = "clae6").drop("clae6", axis=1)
    ## Comercio 5
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio5 , 
                          how = "left", left_on = "actividad5", right_on = "clae6").drop("clae6", axis=1)
    ## Comercio 6
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio6 , 
                          how = "left", left_on = "actividad6", right_on = "clae6").drop("clae6", axis=1)
    return  impo17_bec_complete   

# =============================================================================
# clasificación via destinación
# =============================================================================

    
def metrica(x):
    return (x["valor"] * x["kilos"])/x["cant_decl"]
    
def metrica_(x):    
    if x["cant_decl"] != "Kilogramo":
        divisor = x["cant_decl"]
    elif x["cant_est"] != "Kilogramo":
        divisor = x["cant_est"]
    else: 
        divisor = x["cant_decl"]

    rtn = (x["valor"] * x["kilos"])/divisor
    return rtn

def mod_z(col: pd.Series, thresh: float=3.5):
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = 0.645* ((col - med_col) / med_abs_dev)
    # mod_z = mod_z[np.abs(mod_z) < thresh]
    return np.abs(mod_z)




# =============================================================================
# ordenamiento de los datos de modelo
# =============================================================================
def preprocesamiento_datos(datos, dic_propio,  export_cuit = False):
    
    datos.drop("prob_bk", 1, inplace=True)

    #para pasar el script 1
    datos = diccionario_especial(datos, dic_propio) 
    datos = def_actividades(datos)
    ########

    datos_2trans_1 = datos[ (datos["act_ordenadas"].str.contains("G") ) & (datos["act_ordenadas"].str.contains("J|I|O|K")) ] # CLAE: K|O|H|J|N ---> O coincide entre CLAE e CIIU, J = K, I = H|J|N
    datos_2trans_1 = datos_2trans_1[(datos_2trans_1["act_ordenadas"].str.contains("A|B|C|D|E|F|H|L|M|N|P|Q|R|P|S|T|U"))   ]
    #datos_2trans_2 = datos[ (datos["act_ordenadas"].str.contains("G") ) & (datos["actividades"].str.contains("K_70|J"))]
    datos_ok = datos[~datos.index.isin(datos_2trans_1.index)]
    print("Está ok el split?", len(datos_ok) + len(datos_2trans_1) == len(datos))

    datos_2trans_2 = datos_ok[datos_ok["actividades"].str.contains("K_70|J")]
    datos_2trans_2 = datos_2trans_2[datos_2trans_2["actividades"].str.contains("A|B|C|D|E|F|G|H|I|L|M|N|O|P|Q|R|P|S|T|U|K_71|K_72|K_73|K_74|K_99")]
    datos_ok_2 = datos_ok[~datos_ok.index.isin(datos_2trans_2.index)]
    print("Está ok el split?", len(datos_ok_2) + len(datos_2trans_1) + len(datos_2trans_2)  == len(datos))

    datos_2trans = pd.concat([datos_2trans_1, datos_2trans_2], 0, ignore_index=True)#.drop_duplicates()
    print("Está ok el split?", len(datos_ok_2) + len(datos_2trans) == len(datos))

       # x= datos_2trans[datos_2trans["letra3"]=="J"]
    for letra_i, act_i in zip(["letra1", "letra2", "letra3", "letra4", "letra5", "letra6"],
                              ["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6"]):
        
        letritas = ["J", "I_60", "I_61", "I_62",  "I_63", "I_64", "O", "K_70","K_71", "K_74"]
        act_2change = dic_propio[dic_propio["propio_letra_2"].isin(letritas)]["propio"]
        datos_2trans[act_i]  = np.where(datos_2trans[act_i].isin(act_2change), "999999" , datos_2trans[act_i])

        for letrita in letritas :
            datos_2trans.loc[:, letra_i] = datos_2trans[letra_i].astype(str)
            datos_2trans.loc[:, letra_i] = datos_2trans[letra_i].apply(lambda x: x.replace(letrita, "G"))
    
    rtr = pd.concat([datos_2trans, datos_ok_2 ], axis = 0)
    
    if export_cuit == True:
        datos[["cuit", "NOMBRE"]].drop_duplicates().to_csv("../data/resultados/cuits_unicos.csv", index = False)
        
    return rtr


def asignacion_stp_BK(datos, dic_stp): # input: all data; output: BK
    datos_bk = datos[datos["ue_dest"]== "BK"]
    
    ncm_trans = [
                870421, 870431, #pick-ups confirmado por Maito
                 870422, 870423, 870490, 870432, # camiones
                 870210, 870210,870290, # transporte de pasajeros
                 870120, 870130, 870190 # Tractores de carretera para semirremolques 
                 # 870310, 870321, 870322, 870323, 870324, 870331, 870332,
                 # 870333, 870390 # mandar a consumo
                 ]

    data_trans = datos_bk[datos_bk["HS6"].isin(ncm_trans)].reset_index(drop = True)
    
    ncm_agro = dic_stp[dic_stp["demanda"].str.contains("agríc", case =False)]["NCM"]
    data_agro = datos_bk[datos_bk["HS6"].isin(ncm_agro)].reset_index(drop = True)
    
    datos_bk_filtro = datos_bk[~(datos_bk["HS6"].isin(ncm_trans)) & ~(datos_bk["HS6"].isin(ncm_agro))]

    ncm_j = [880230, 842720, 901813, 491199, 845610, 842710, 850220,
             842720, 847150, 845710] #mierdas que caen en intermediación financiera
    data_financiera = datos_bk[datos_bk["HS6"].isin(ncm_j)].reset_index(drop = True)

    ncm_k_70 = [901841, 843050, 842839, 860400, 901849, 842641, 940510,
                847910, 847432, 846592]
    data_inmobiliaria = datos_bk[datos_bk["HS6"].isin(ncm_k_70)].reset_index(drop = True)

    
    letras = ["letra1", "letra2", "letra3","letra4", "letra5", "letra6"]
    for letra in letras:
        data_trans[letra] = data_trans[letra].replace(regex=[r'^D.*$'],value="I_60") #con regex, buscar que empiece con D_ y poner I_60
        data_trans[letra] = data_trans[letra].replace(["G","K_70", "J"] , "I_60") #CIIU => K: Act inmobiliarias y empresariales // J: intermediacion financiera
    
    for letra in letras:
        data_agro[letra] = data_agro[letra].replace(regex=[r'^D.*$'],value="A") #con regex, buscar que empiece con D_ y poner A
        data_agro[letra] = data_agro[letra].replace(["G","K_70", "J"] , "A")

    for letra in letras:
        data_financiera[letra] = data_financiera[letra].replace(["J"], "G")

    for letra in letras:
    data_inmobiliaria[letra] = data_financiera[letra].replace(["K_70"], "G")
    
    datos_bk = pd.concat([datos_bk_filtro, data_trans, data_agro, data_financiera, data_inmobiliaria], axis=0)
    
    datos_bk_sin_picks = datos_bk[~datos_bk["HS6"].isin([870421, 870431])]
    bk_picks = datos_bk[datos_bk["HS6"].isin([870421, 870431])]
    
    return datos_bk, datos_bk_sin_picks, bk_picks



def filtro_ci(datos):
    import datetime
    start = datetime.datetime.now()
    datos_ci= datos[datos["ue_dest"]== "CI"]
    
    # datos_2trans = datos_ci[ (datos_ci["act_ordenadas"].str.contains("G") ) & (datos_ci["act_ordenadas"].str.contains("J|I|O") ) ] # CLAE: K|O|H|J|N ---> O coincide entre CLAE e CIIU, J = K, I = H|J|N
    # datos_ok= datos_ci[ ~datos_ci.index.isin(datos_2trans.index) ]
    # print("Está ok el split?", len(datos_ok)+ len(datos_2trans) == len(datos_ci))
    
    # # x= datos_2trans[datos_2trans["letra3"]=="J"]
    # for letra_i, act_i in zip(["letra1", "letra2", "letra3", "letra4", "letra5", "letra6"],
    #                           ["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6"]):
    #     for letrita in ["J", "I_60", "I_61", "I_62", "I_64", "O"]:
    #         datos_2trans.loc[:, letra_i] = datos_2trans[letra_i].apply(lambda x: x.replace(letrita, "G"))
    # rtr = pd.concat([datos_2trans, datos_ok ], axis = 0)
    
    end = datetime.datetime.now()
    print(end-start)
    return datos_ci




   


################################
# CLASIFICACION TIPO BIEN
################################

def fun_filtro_stp(dic_stp, impo_bec):
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
    ya_filtrado = impo_bec[impo_bec["ue_dest"] != ""]

    print( "Los datos asignados y sin asignar poseen mismo largo que los BK?", (len(filtro1) + len(ya_filtrado)) == len(impo_bec))

    return  filtro1,ya_filtrado, impo_bec

def clasif_diag(impo_bec, tipo_bien):
    impo_bec_ci = impo_bec[impo_bec["BEC5EndUse"].str.startswith(tipo_bien, na=False)]
  
    filtro1st = impo_bec_ci[impo_bec_ci["dest_clean"] == "S/TR"]
    filtro1ct = impo_bec_ci[impo_bec_ci["dest_clean"] == "C/TR"]
    filtro1co = impo_bec_ci[impo_bec_ci["dest_clean"] == "CONS&Otros"]

    print( "Archivos con igual longitud?", len(impo_bec_ci) == (len(filtro1st) + len(filtro1ct) + len(filtro1co)) )

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

    print("consistencia de diagramas 2", (len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) + len(dest_b) + len(dest_c)) == len(impo_bec_ci) )

    dest_a["filtro"] = "A"
    dest_b["filtro"] = "B"
    dest_c["filtro"] = "C"
    dest_d["filtro"] = "D"
    dest_e["filtro"] = "E"
    dest_f["filtro"] = "F"
    dest_g["filtro"] = "G"

    data_post = pd.concat([dest_a, dest_b, dest_c, dest_d, dest_e, dest_f, dest_g], axis=0)
    return data_post, dest_c


def clasificacion_BK(filtro1):
    filtro1, dest_c = clasif_diag(filtro1, "CAP")

    # filtro1["ue_dest"].value_counts()
    filtro1["ue_dest"] = np.where((filtro1["filtro"] == "A") &
                                    (filtro1["dest_clean"] == "S/TR"),
                                        "BK",
                                        np.where( (filtro1["filtro"] == "B") &
                                                (filtro1["dest_clean"] == "C/TR"),
                                                "CI", ""
                                                )
                                    )

    # filtro1[["ue_dest", "filtro"]].value_counts()


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
 
    filtro2_G_clas = filtro2[(filtro2["filtro"]=="G")  & (filtro2["dest_clean"]!="CONS&Otros")]
    filtro2_G_clas ["ue_dest"] = np.where(filtro2_G_clas["dest_clean"]== "S/TR","BK", "CI" )
  
    clasif_G = pd.concat([filtro2_G_clas, filtro2_G.drop(["median_bk", "median_ci",  "mad_bk", "mad_ci", "z_score_bk", "z_score_ci"],axis =1 )], axis = 0)
   
    ## EDA de Datos ya clasificados
    clasif_AB = filtro1[filtro1["ue_dest"] != "" ]
    clasif_AB["metric"] = clasif_AB.apply(lambda x: metrica(x), axis = 1)

    data_clasif = pd.concat([clasif_AB, clasif_D ,clasif_E,clasif_G  ], axis= 0)
    # data_clasif["metric_zscore"] = (data_clasif["metric"] -data_clasif["metric"].mean())/ data_clasif["metric"].std(ddof=1)

    # data_clasif[["filtro", "ue_dest"]].value_counts()

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
    
    stp_ue_dest ["metric"] = stp_ue_dest .apply(lambda x: metrica(x), axis = 1)
    stp_ue_dest["filtro"] = "STP"

    data_clasif_ue_dest = pd.concat([data_clasif, stp_ue_dest], axis = 0)


    data_clasif_ue_dest["precio_kilo"]= data_clasif_ue_dest["valor"]/data_clasif_ue_dest["kilos"]

    data_clasif_ue_dest.to_csv("../data/resultados/bk_con_ue_dest.csv")

    return  data_clasif_ue_dest


def clasificacion_CI(impo_bec):
    cons_int , dest_c= clasif_diag(impo_bec, "INT")
    impo_bec_ci = impo_bec[impo_bec["BEC5EndUse"].str.startswith("INT", na=False)]

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

    cons_int_G_no_cons = cons_int[(cons_int["filtro"] == "G") & (cons_int["dest_clean"] != "CONS&Otros")]
    cons_int_G_no_cons["ue_dest"] = np.where(cons_int_G_no_cons["dest_clean"] == "S/TR", "BK", "CI")
    # cons_int_G_clas["ue_dest"].value_counts()

    cons_int_G_clasif = pd.concat([cons_int_G_no_cons, cons_int_G.drop(
        ["median_bk", "median_ci", "mad_bk", "mad_ci", "z_score_bk", "z_score_ci"], axis=1)], axis=0)
   
    cons_int_clasif = pd.concat([cons_int_D, cons_int_G_clasif, cons_int[cons_int["filtro"].str.contains("A|B|C|E")]])

    return cons_int_clasif ,impo_bec_ci


def clasificacion_CONS(impo_bec):
    cons_fin , dest_c= clasif_diag(impo_bec, "CONS")
    impo_bec_cons = impo_bec[impo_bec["BEC5EndUse"].str.startswith("CONS", na=False)]

    # A, B, C, E
    cons_fin["ue_dest"] = np.where((cons_fin["filtro"] == "B") |
                                   (cons_fin["filtro"] == "E") |
                                   (cons_fin["filtro"] == "C"), "CI",
                                   np.where((cons_fin["filtro"] == "A"), "BK",
                                            np.nan))

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

def def_actividades(data_model):
    data_model["actividades"] = data_model["letra1"] + data_model["letra2"] + data_model["letra3"] + data_model[
        "letra4"] + data_model["letra5"] + data_model["letra6"]
    data_model["actividades_ordenadas"] = data_model["actividades"].str.replace('\d+|_', '')
    data_model["act_ordenadas"] = data_model["actividades_ordenadas"].apply( lambda x: "".join(sorted(x)))  # "".join(sorted(data_model["actividades"]))
    return data_model.drop("actividades_ordenadas", 1)


def concatenacion_ue_dest(cons_fin_clasif, cons_int_clasif,data_clasif_ue_dest,data_not_clasif, join_impo_clae, impo_bec):
    # impo_ue_dest = pd.concat([pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha", 'metric', 'ue_dest', 'mad', 'median', 'z_score'], axis = 1), bk], axis =0)
    cicf_ue_dest = pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha",  'mad', 'median', 'z_score'], axis = 1) #, bk], axis =0)
    cicf_ue_dest["precio_kilo"] =  cicf_ue_dest["valor"]/cicf_ue_dest["kilos"]

    bk_ue_dest = data_clasif_ue_dest#.copy().drop(['HS4', 'HS4Desc', 'HS6Desc', "BEC5Category"], 1)
    bk_sin_ue_dest = data_not_clasif#.drop(['HS4', 'HS4Desc', 'HS6Desc', "BEC5Category"], 1)

    print("registros faltantes?", (len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()])) - (len(bk_sin_ue_dest) + len(bk_ue_dest) + len(cicf_ue_dest)))
    data_model = pd.concat([bk_sin_ue_dest , bk_ue_dest, cicf_ue_dest ], axis = 0) #los datos sin clasificar quedan arriba del dataset

    #PREPROCESAMIENTO
    data_model ['HS6'] = data_model ['HS6'].astype("str")
    data_model ['HS8'] = data_model ['HS6_d12'].str.slice(0,8)
    data_model ['HS10'] = data_model ['HS6_d12'].str.slice(0,10)
    data_model = def_act_ordenadas(data_model)

    # preprocesamiento etiquetados
    cols = ["cuit", "NOMBRE",
            "HS6", "HS8", "HS10", "HS6_d12",
            "uni_est", "cant_est", "uni_decl", "cant_decl",
            'valor', 'kilos', "precio_kilo", "metric",
            "letra1", "letra2", "letra3", "letra4", "letra5", "letra6",
            "actividad1","actividad2","actividad3","actividad4","actividad5","actividad6",
            "act_ordenadas",
            "dest_clean", "filtro", "BEC5EndUse",
            "ue_dest"]

    data_model = data_model[cols]

    return data_model


def str_a_num(df):

  for  (columnName, columnData)  in df.iteritems():
    
    original = np.sort(np.unique(columnData.values ))
    reemplazo = list(range(len(original))) # lo dejo como lista pero creo q se podria sacar
    mapa = dict(zip(original, reemplazo))
    
    df[ columnName] = df[columnName].map(mapa)
  return df

def predo_datos_modelo(data_model):
   # Filtros de columnas
    cols_reservadas = ["cuit", "NOMBRE", "HS6_d12",  "dest_clean",
                       "actividad1","actividad2","actividad3","actividad4","actividad5","actividad6",
                     # "act_ordenadas",
                       "uni_est", "uni_decl", "BEC5EndUse", "filtro" ,
                       "ue_dest" ]
    # cat_col = list(data_model.select_dtypes(include=['object']).columns)
    # cat_col = [elem for elem in cat_col if elem not in cols_reservadas ]
    
    labeled_cols = ["HS6", "HS8", "HS10", "act_ordenadas",  'letra1','letra2', 'letra3', 'letra4', 'letra5', 'letra6']
    ohe_cols= ["BEC5EndUse", "dest_clean",  'letra1','letra2', 'letra3', 'letra4', 'letra5', 'letra6']
    
    num_col = list(data_model.select_dtypes(include=['float', "int64"]).columns)
    num_col = [elem for elem in num_col if elem != "cuit"]

    one_hot = pd.get_dummies(data_model[ohe_cols])
    data_pre = pd.concat( [ str_a_num(data_model[labeled_cols]) ,  data_model[num_col], one_hot, data_model["ue_dest"] ], axis = 1  )

    # datos etiquetados
    data_train = data_pre[data_pre ["ue_dest"] != "?" ]
    data_train["bk_dummy"] = data_train["ue_dest"].map({"BK": 1, "CI": 0})
    data_train.drop("ue_dest", axis = 1, inplace = True)

    # datos no etiquetados
    data_to_clasif = data_pre[data_pre["ue_dest"] == "?" ]
    data_to_clasif.drop("ue_dest", axis = 1, inplace = True)

    print(len(data_pre) == (len(data_train) + len(data_to_clasif)))

    return data_pre, data_train, data_to_clasif

def predo_datos_modelo_21oct(data_model):
    # Filtros de columnas
    cols_reservadas = ["cuit", "NOMBRE", "HS6_d12",  "dest_clean",
                       "actividad1","actividad2","actividad3","actividad4","actividad5","actividad6",
                     # "act_ordenadas",
                       "uni_est", "uni_decl", "BEC5EndUse", "filtro" ,
                       "ue_dest" ]
    # cat_col = list(data_model.select_dtypes(include=['object']).columns)
    # cat_col = [elem for elem in cat_col if elem not in cols_reservadas ]
    
    labeled_cols = ["HS6", "HS8", "HS10", "act_ordenadas",  'letra1','letra2', 'letra3', 'letra4', 'letra5', 'letra6', "uni_est", "uni_decl"]
    ohe_cols= ["BEC5EndUse", "dest_clean",  'letra1','letra2', 'letra3', 'letra4', 'letra5', 'letra6']
    
    num_col = [  'valor',  'kilos', "precio_kilo" , "metric", "cant_est", "cant_decl"]

    data_pre = pd.concat( [ str_a_num(data_model[labeled_cols]) ,  data_model[num_col], data_model["ue_dest"] ], axis = 1  )

    # datos etiquetados
    data_train = data_pre[data_pre ["ue_dest"] != "?" ]
    data_train["bk_dummy"] = data_train["ue_dest"].map({"BK": 1, "CI": 0})
    data_train.drop("ue_dest", axis = 1, inplace = True)

    # datos no etiquetados
    data_to_clasif = data_pre[data_pre["ue_dest"] == "?" ]
    data_to_clasif.drop("ue_dest", axis = 1, inplace = True)

    print(len(data_pre) == (len(data_train) + len(data_to_clasif)))

    return data_pre, data_train, data_to_clasif
