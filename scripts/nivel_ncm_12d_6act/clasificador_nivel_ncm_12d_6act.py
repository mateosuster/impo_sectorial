# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:54 2021

@author: igalk
"""

# =============================================================================
# Directorio de trabajo y librerias
# =============================================================================
import os 
#Mateo
os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
#igal
# os.chdir("C:/Users/igalk/OneDrive/Documentos/laburo/CEP/procesamiento impo/nuevo1/impo_sectorial/scripts/nivel_ncm_12d_6act")

# os.getcwd()
import pandas as pd
import numpy as np
import re
import tqdm
import datatable as dt

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from nivel_ncm_12d_6act.Bases_nivel_ncm_12d_6act import *
from nivel_ncm_12d_6act.procesamiento_nivel_ncm_12d_6act import *
from nivel_ncm_12d_6act.matriz_nivel_ncm_12d_6act import *
from nivel_ncm_12d_6act.pre_visualizacion_nivel_ncm_12d_6act import *




#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
#datasetes utilizados en ultimas corridas 
impo_d12 = pd.read_csv("../data/impo_2017_diaria.csv")
clae = pd.read_csv( "../data/clae_nombre.csv")
comercio = pd.read_csv("../data/comercio_clae.csv", encoding="latin1")
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")
bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018_HS12.csv", sep = ";")
ncm12_desc = pd.read_csv("../data/d12_2012-2017.csv", sep=";")
dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")
data_predichos = pd.read_csv("../data/resultados/datos_clasificados_modelo_all_data.csv", sep = ";").drop("Unnamed: 0", 1)# output del modelo
datos_clasificados = pd.read_csv("../data/resultados/data_modelo_diaria.csv")
# CIIU
dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")
clae_to_ciiu = pd.read_excel("../data/Pasar de CLAE6 a CIIU3.xlsx")
hs_to_isic = pd.read_csv("../data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )


#impo 12 d
# impo_d12 = pd.read_csv("../data/IMPO_17_feature.csv")
# impo_d12 = pd.read_csv("../data/IMPO_2017_12d.csv")
#impo_17 = pd.read_csv(  "../data/IMPO_2017.csv", sep=";")

#cuit_clae = pd.read_csv( "../data/cuit 2017 impo_con_actividad.csv")
# bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018.csv")
# bec_to_clae = pd.read_csv("../data/bec_to_clae.csv")
#diccionario ncm12d
# ncm12_desc = pd.read_csv("../data/NCM 12d.csv", sep=";")
# ncm12_desc_split = pd.concat([ncm12_desc.iloc[:,0], pd.DataFrame(ncm12_desc['Descripción Completa'].str.split('//', expand=True))], axis=1)

# parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")

# bce_cambiario = pd.read_csv("../data/balance_cambiario.csv", skiprows = 3, error_bad_lines=False, sep= ";", na_values =['-'])


#############################################
#           preparación bases               #
#############################################
# predo_impo_17(impo_17)
ncm12_desc_mod = predo_ncm12_desc(ncm12_desc )["ncm_desc"]    
impo_d12  = predo_impo_12d(impo_d12, ncm12_desc_mod )
letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae)
bec_bk = predo_bec_bk(bec)#, bec_to_clae)
dic_stp = predo_stp(dic_stp)
datos = predo_datamodel(data_predichos, datos_clasificados )
ciiu_dig_let = predo_ciiu(clae_to_ciiu, dic_ciiu)

# preprocesamiento CIIU
# dic_ciiu = predo_dic_ciiu(dic_ciiu)
# ciiu_letra = predo_ciiu_letra(dic_ciiu, comercio)




#############################################
#                joins:                     ESTA PARTE ES INNCESARIA #
#############################################
# join_impo_clae = def_join_impo_clae(impo_d12, cuit_empresas) #join CUIT
# join_impo_clae_bec_bk = def_join_impo_clae_bec(join_impo_clae, bec_bk) # filtro BK

############################################################
#  Asignación por STP / modificación de actividades x ncm  #
############################################################
#revisión de nulos de BEC5EndUSE
# impo_bec = pd.merge(join_impo_clae, bec[["HS6", "BEC5EndUse" ]], how= "left" , left_on = "HS6", right_on= "HS6" )
# (len(datos ) + len(impo_bec[impo_bec["BEC5EndUse"].isnull()]) ) == len(join_impo_clae)


###aca igal metio un cambio de orden!!

#datos_bk = diccionario_especial(datos_bk,ciiu_dig_let) ###### viernes a la noche. cambié el orden de estas lineas

datos_bk = diccionario_especial(datos,ciiu_dig_let) #cambio igal
letras_mod = letra_nn(datos_bk) # obtencion de LETRA_nn
datos_bk = pd.concat([datos_bk.drop( ["letra1","letra2","letra3","letra4", "letra5", "letra6"], axis = 1),  letras_mod ], axis = 1)
# datos_bk.to_csv("../data/resultados/importaciones_bk_pre_intro_matriz.csv")
datos_bk = asignacion_stp_BK(datos_bk, dic_stp)
# hasta aca!!

#############################################
#           Tabla de contingencia           #
#              producto-sector              #
#############################################
join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(datos_bk , comercio) 

tabla_contingencia = def_contingencia(join_impo_clae_bec_bk_comercio)

#############################################
#      ponderación por ncm y letra          #
#############################################
join_impo_clae_bec_bk_comercio_pond = def_join_impo_clae_bec_bk_comercio_pond(join_impo_clae_bec_bk_comercio, tabla_contingencia)

join_final = def_calc_pond(join_impo_clae_bec_bk_comercio_pond,tabla_contingencia)
#join_final.to_csv("../data/resultados/impo_con_ponderaciones_12d_6act_post_ml.csv", index=False)
#join_final = pd.read_csv("../data/resultados/impo_con_ponderaciones_12d_6act_post_ml.csv")

#############################################
#         ASIGNACIÓN y MATRIZ               #
#############################################
matriz_sisd_insumo = def_insumo_matriz(join_final)
# matriz_sisd_insumo.to_csv("../data/resultados/matriz_pesada_12d_6act_postML.csv", index= False)
#matriz_sisd_insumo= pd.read_csv("../data/resultados/matriz_pesada_12d_6act_postML.csv")

#asignación por probabilidad de G-bk (insumo para la matriz)
asign_pre_matriz= def_matriz_c_prob(matriz_sisd_insumo)
# asign_pre_matriz.to_csv("../data/resultados/asign_pre_matriz.csv")

#matriz SISD
matriz_sisd = to_matriz(asign_pre_matriz)
matriz_hssd  = pd.pivot_table(asign_pre_matriz, values='valor_pond', index=['hs6_d12'], columns=['sd'], aggfunc=np.sum, fill_value=0) 
# matriz_sisd.to_csv("../data/resultados/matriz_sisd.csv")

#filtro para destinación de productos
# x = matriz_hssd[matriz_hssd.index.str.startswith(("870421", "870431"))]
# x.sum(axis = 0)
# matriz_sisd.sum().sum()
# =============================================================================
#                       Visualizacion
# =============================================================================
#preprocesamiento
sectores_desc = sectores() #Sectores CLAE
letras_ciiu = pd.DataFrame(matriz_sisd.index)
impo_tot_sec = impo_total(z = matriz_sisd, sectores_desc= False) 
comercio_y_propio = impo_comercio_y_propio(matriz_sisd, sectores_desc = False) 

# armo nuevo diccionario
# letras_ciiu_pd =  dic_ciiu[["ciiu3_letra", "ciiu3_letra_desc"]].drop_duplicates(subset = "ciiu3_letra")
# letras_ciiu_pd [letras_ciiu_pd["ciiu3_letra"].str.contains("D", "I", "K" , na = False) ]
# letras_ciiu_pd [letras_ciiu_pd["ciiu3_letra"].str.contains("D") ]

# letras_ciiu_pd.info()
# dic_ciiu.info()
# x = dic_ciiu[dic_ciiu["ciiu3_2c"].astype(str).str.startswith("34")]


dic_ciiu[dic_ciiu["ciiu3_2c"].astype(str).str.startswith("34")]

digitos = tuple(list(map(str, range(15, 38))) + list(map(str, range(60, 65))) )
ciiu_desc_2= dic_ciiu[dic_ciiu["ciiu3_2c"].astype(str).str.startswith(digitos )][[ "ciiu3_letra", "ciiu3_2c", "ciiu3_2c_desc" ]].drop_duplicates()
ciiu_desc_2["ciiu3_2c"] =  ciiu_desc_2["ciiu3_letra"] + "_"+ ciiu_desc_2["ciiu3_2c"].astype(str).str.slice(0,2) 

ciiu_desc_3= dic_ciiu[dic_ciiu["ciiu3_3c"].astype(str).str.startswith( ("551", "552"), na = False)][[ "ciiu3_letra", "ciiu3_3c", "ciiu3_3c_desc"  ]].drop_duplicates()
ciiu_desc_3["ciiu3_3c"] =  ciiu_desc_3["ciiu3_letra"] + "_"+ ciiu_desc_3["ciiu3_3c"].astype(str).str.slice(0,3) 


ciiu_desc.info()

dic_ciiu.info()

# graficos
graficos(matriz_sisd, impo_tot_sec, comercio_y_propio)

##### tabla top 5
# Top 5 de importaciones de cada sector
top_5_impo = top_5(asign_pre_matriz, letras_ciiu , ncm12_desc_mod, impo_tot_sec) # a veces rompe por la var HS12, pero se soluciona corriendo de nuevo el preprocesamiento
top_5_impo.to_excel("../data/resultados/top5_impo.xlsx")

#CUITS QUE IMPORTAN TOP HS6 Industria
# top_industria = top_5_impo[top_5_impo["letra"]=="C"]["hs6"].iloc[[0,3]]
# cuit_top_c = join_impo_clae[join_impo_clae["HS6"].isin(top_industria )].sort_values("valor",ascending=False)#["CUIT_IMPOR"].unique()
# cuit_empresas[cuit_empresas["cuit"].isin(cuit_top_c)]




