# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:06:12 2021

@author: igalk
"""

import os 
os.getcwd()

import pandas as pd
import numpy as np


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
    return comercio_reclasificado

def predo_cuit_clae(cuit_clae, tipo_cuit):
    
    cuit_clae6 = cuit_clae[cuit_clae["Numero_actividad_cuit"]<=6].drop(["Clae6_desc","Fecha_actividad", "Cantidad_actividades_cuit"], axis =1)

    cuit_clae6 = cuit_clae6.set_index(["CUIT", "Numero_actividad_cuit"], append=True)

    cuit_clae6 = cuit_clae6.unstack().droplevel(0, axis=1)
    
    cuit_clae6 = cuit_clae6.groupby("CUIT").sum().astype(int)
    
    cuit_clae6.columns = ["actividad1","actividad2","actividad3","actividad4","actividad5","actividad6"]
    
    cuit_clae_letra = pd.merge(cuit_clae, clae[["clae6", "letra"]], left_on="Clae6", right_on= "clae6", how = "left").drop(["clae6", "Clae6"], axis =1)
    
    cuit_clae_letra = cuit_clae_letra.groupby("CUIT")["letra"].agg(",".join)
    
    cuit_clae_letra = pd.DataFrame(cuit_clae_letra.groupby("CUIT")["letra"].agg(",".join)).reset_index()
    
    cuit_clae_letra["letra"].str.split(pat=",", expand=True)
   
    

    
    cuit_personas = cuit_clae[(cuit_clae["letra1"].isnull()) & (cuit_clae["letra2"].isnull()) & (cuit_clae["letra3"].isnull())]
    cuit_empresas = cuit_clae[~cuit_clae['cuit'].isin(cuit_personas['cuit'])]
    
    
    if tipo_cuit == "personas":
        return cuit_personas 
        
    else:
        #reemplzao faltantes de letra1 con letra2, y faltantes letra 2 con letra 3
        cuit_empresas['letra1'].fillna(cuit_empresas['letra2'], inplace = True) 
        cuit_empresas['letra2'].fillna(cuit_empresas['letra3'], inplace = True) #innecesario, porque no cambia la cuenta (quien tiene NaN en letra 2 tambien tiene NaN letra 3)
        
        #completo relleno de faltantes con letra 1
        cuit_empresas['letra2'].isnull().sum()
        cuit_empresas['letra3'].isnull().sum()
        cuit_empresas['letra2'].fillna(cuit_empresas['letra1'], inplace = True) 
        cuit_empresas['letra3'].fillna(cuit_empresas['letra1'], inplace = True) 

        cuit_clae_letra = cuit_empresas.drop(["padron_contribuyentes","actividad_mectra", "letra_mectra"], axis = 1 , inplace = False)
                
        return cuit_clae_letra
    
def predo_bec_bk(bec, bec_to_clae):
    bec_cap = bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]
    #partes y accesorios dentro de start with cap
    partes_accesorios  = bec_cap[bec_cap["HS6Desc"].str.contains("part|acces")]   
    partes_accesorios["BEC5EndUse"].value_counts()

    # filtro bienes de capital
    bec_bk = bec_cap.loc[~bec_cap["HS6"].isin( partes_accesorios["HS6"])]
    
    bec_to_clae.drop(bec_to_clae.iloc[:, 2:], axis = 1, inplace = True)
    bec_to_clae = bec_to_clae.rename(columns={"BEC Category": "BEC5Category"})
    bec_to_clae["bec_to_clae"] = bec_to_clae["bec_to_clae"].str.replace(",", "").str.replace(" ", "")  

    filtro = [ "HS4", "HS4Desc", "HS6", "HS6Desc", "BEC5Category","BEC5Specification", "BEC5EndUse",
              "BEC4Code", 'BEC4ENDUSE', 'BEC4INT', 'BEC4CONS', 'BEC4CAP' ]
    
    bec_bk = pd.merge(bec_bk[filtro], bec_to_clae , left_on= "BEC5Category", right_on = "BEC5Category", how= "left")
    
    return bec_bk

def def_join_impo_clae(impo_17, cuit_empresas):
    impo_clae = pd.merge(impo_17, cuit_empresas, left_on = "CUIT_IMPOR", right_on = "cuit", how = "right")
    impo_clae.drop(["cuit", "denominacion"], axis=1, inplace = True)
       
    return impo_clae

def def_join_impo_clae_bec(join_impo_clae, bec_bk):
    impo17_bec_bk = pd.merge(join_impo_clae, bec_bk, how= "left" , left_on = "HS6", right_on= "HS6" )

    # filtramos las impos que no mergearon (no arrancan con CAP)
    impo17_bec_bk = impo17_bec_bk[impo17_bec_bk["HS4"].notnull()]

    return impo17_bec_bk

    

def def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio):
    comercio2 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = { "vta_vehiculos":"vta_vehiculos2",
                                                                                   "vta_bk": "vta_bk2", "vta_sec": "vta_sec2"})

    comercio3 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = { "vta_vehiculos":"vta_vehiculos3",
                                                                                   "vta_bk": "vta_bk3", "vta_sec": "vta_sec3"})

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

    return  impo17_bec_complete   

    