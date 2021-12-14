# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:54 2021

@author: Igal Kejsefman y Mateo Suster
"""

# =============================================================================
# Directorio de trabajo y librerias
# =============================================================================
import os 
#os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
# os.chdir("C:/Users/igalk/OneDrive/Documentos/laburo/CEP/procesamiento impo/nuevo1/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")

# os.getcwd()
import pandas as pd
import numpy as np
import re
import tqdm
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from Bases_nivel_ncm_12d_6act import *
from procesamiento_nivel_ncm_12d_6act import *
from matriz_nivel_ncm_12d_6act import *
from pre_visualizacion_nivel_ncm_12d_6act import *



#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
#impo_d12 = pd.read_csv("../data/impo_2017_diaria.csv") #falta
clae = pd.read_csv( "../data/clae_nombre.csv")
comercio = pd.read_csv("../data/comercio_clae.csv", encoding="latin1")
comercio_ci = pd.read_csv("../data/vector_de_comercio_clae_ci.csv", sep = ";",encoding="utf-8")
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")
bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018_HS12.csv", sep = ";")
ncm12_desc = pd.read_csv("../data/d12_2012-2017.csv", sep=";")
dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")
data_predichos = pd.read_csv("../data/resultados/datos_clasificados_modelo_all_data.csv", sep = ";")
datos_clasificados = pd.read_csv("../data/resultados/data_modelo_diaria.csv") #falta

# CIIU
dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")
clae_to_ciiu = pd.read_excel("../data/Pasar de CLAE6 a CIIU3.xlsx")
hs_to_isic = pd.read_csv("../data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )


#############################################
#           preparación bases               #
#############################################
# predo_impo_17(impo_17)
ncm12_desc_mod = predo_ncm12_desc(ncm12_desc )["ncm_desc"]    
#impo_d12  = predo_impo_12d(impo_d12, ncm12_desc_mod )
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


############################################################
#  Asignación por STP / modificación de actividades x ncm  #
############################################################
#revisión de nulos de BEC5EndUSE
# impo_bec = pd.merge(join_impo_clae, bec[["HS6", "BEC5EndUse" ]], how= "left" , left_on = "HS6", right_on= "HS6" )
# (len(datos ) + len(impo_bec[impo_bec["BEC5EndUse"].isnull()]) ) == len(join_impo_clae)

datos = diccionario_especial(datos,ciiu_dig_let) #cambio igal
letras_mod = letra_nn(datos) # obtencion de LETRA_nn
datos = pd.concat([datos.drop( ["letra1","letra2","letra3","letra4", "letra5", "letra6"], axis = 1),  letras_mod ], axis = 1)
# datos.to_csv("../data/resultados/importaciones_bk_pre_intro_matriz.csv")

datos_bk = asignacion_stp_BK(datos, dic_stp)
datos_ci = filtro_ci(datos)

# cuits importadores unicos
pd.DataFrame(datos.cuit.unique(), columns = ["cuits"]).to_csv("../data/resultados/cuits_importadores_unicos.csv")

#############################################
#           Tabla de contingencia           #
#              producto-sector              #
#############################################
join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(datos_bk , comercio) 
join_impo_clae_bec_ci_comercio = def_join_impo_clae_bec_bk_comercio(datos_ci , comercio_ci, ci = True)

tabla_contingencia = def_contingencia(join_impo_clae_bec_bk_comercio, datos)
tabla_contingencia_ci = def_contingencia(join_impo_clae_bec_ci_comercio)

#############################################
#      ponderación por ncm y letra          #
#############################################
join_impo_clae_bec_bk_comercio_pond = def_join_impo_clae_bec_bk_comercio_pond(join_impo_clae_bec_bk_comercio, tabla_contingencia)
join_impo_clae_bec_ci_comercio_pond = def_join_impo_clae_bec_bk_comercio_pond(join_impo_clae_bec_ci_comercio, tabla_contingencia_ci)

join_final = def_calc_pond(join_impo_clae_bec_bk_comercio_pond,tabla_contingencia, ci = False)
join_final_ci = def_calc_pond(join_impo_clae_bec_ci_comercio_pond, tabla_contingencia_ci, ci = True)
#join_final.to_csv("../data/resultados/impo_con_ponderaciones_12d_6act_post_ml.csv", index=False)
#join_final = pd.read_csv("../data/resultados/impo_con_ponderaciones_12d_6act_post_ml.csv")


#############################################
#         ASIGNACIÓN y MATRIZ               #
#############################################
matriz_sisd_insumo = def_insumo_matriz(join_final)
matriz_sisd_insumo_ci = def_insumo_matriz(join_final_ci, ci = True)
# matriz_sisd_insumo.to_csv("../data/resultados/matriz_pesada_12d_6act_postML.csv", index= False)
#matriz_sisd_insumo= pd.read_csv("../data/resultados/matriz_pesada_12d_6act_postML.csv")

#asignación por probabilidad de G-bk (insumo para la matriz)
asign_pre_matriz= def_matriz_c_prob(matriz_sisd_insumo)
asign_pre_matriz_ci= def_matriz_c_prob(matriz_sisd_insumo_ci)
# asign_pre_matriz.to_csv("../data/resultados/asign_pre_matriz.csv")

#matriz SISD
matriz_sisd = to_matriz(asign_pre_matriz)
matriz_sisd_ci = to_matriz(asign_pre_matriz_ci, ci = True)
# matriz_sisd= pd.read_csv("../data/resultados/matriz_sisd.csv")

matriz_hssd  = pd.pivot_table(asign_pre_matriz, values='valor_pond', index=['hs6_d12'], columns=['sd'], aggfunc=np.sum, fill_value=0) 
matriz_hssd_ci  = pd.pivot_table(asign_pre_matriz_ci, values='valor_pond', index=['hs6_d12'], columns=['sd'], aggfunc=np.sum, fill_value=0) 
# matriz_sisd.to_csv("../data/resultados/matriz_sisd.csv")

#filtro para destinación de productos
# x = matriz_hssd[matriz_hssd.index.str.startswith(("870421", "870431"))]
# x.sum(axis = 0)
matriz_sisd_ci.sum().sum()

# =============================================================================
#                       Otros
# =============================================================================

#exponentes a aplicar a la tabla de contingencia
z = pd.DataFrame(datos["HS6_d12"].value_counts()).reset_index(drop=True)

z['freq'] = z.groupby('HS6_d12')['HS6_d12'].transform('count')

z =  z.drop_duplicates()

z["expo"] = 2+np.log10(z["HS6_d12"])

sns.lineplot(data= z, x= "HS6_d12", y= "expo")



# =============================================================================
#                       Visualizacion
# =============================================================================
#preprocesamiento
sectores_desc = sectores() #Sectores CLAE
letras_ciiu = dic_graf(matriz_sisd_ci, dic_ciiu)
letras_ciiu["desc"] = letras_ciiu["desc"].str.slice(0,15)

impo_tot_sec = impo_total(matriz_sisd, sectores_desc= False, letras_ciiu = letras_ciiu) 
impo_tot_sec_ci = impo_total(matriz_sisd_ci, sectores_desc= False, letras_ciiu = letras_ciiu)
comercio_y_propio = impo_comercio_y_propio(matriz_sisd,letras_ciiu, sectores_desc = False)
comercio_y_propio_ci = impo_comercio_y_propio(matriz_sisd_ci,letras_ciiu, sectores_desc = False)

x = pd.merge(matriz_sisd.reset_index(),letras_ciiu, how = "outer", left_on= "si",  right_on="letra")

# graficos
graficos(matriz_sisd, impo_tot_sec, comercio_y_propio, letras_ciiu, titulo = "Bienes de Capital")
graficos(matriz_sisd_ci, impo_tot_sec_ci, comercio_y_propio_ci, letras_ciiu, titulo = "Consumos Intermedios")

##### tabla top 5
# Top 5 de importaciones de cada sector
top_5_impo = top_5(asign_pre_matriz, letras_ciiu , ncm12_desc_mod, impo_tot_sec) # a veces rompe por la var HS12, pero se soluciona corriendo de nuevo el preprocesamiento
top_5_impo.to_excel("../data/resultados/top5_impo.xlsx")

top_5_impo_ci = top_5(asign_pre_matriz_ci, letras_ciiu , ncm12_desc_mod, impo_tot_sec_ci) # a veces rompe por la var HS12, pero se soluciona corriendo de nuevo el preprocesamiento
top_5_impo_ci.to_excel("../data/resultados/top5_impo_ci.xlsx")


asign_pre_matriz_ci[asign_pre_matriz_ci.sd == "P"]

#CUITS QUE IMPORTAN TOP HS6 Industria
# top_industria = top_5_impo[top_5_impo["letra"]=="C"]["hs6"].iloc[[0,3]]
# cuit_top_c = join_impo_clae[join_impo_clae["HS6"].isin(top_industria )].sort_values("valor",ascending=False)#["CUIT_IMPOR"].unique()
# cuit_empresas[cuit_empresas["cuit"].isin(cuit_top_c)]



