# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:54 2021

@author: igalk
"""

# =============================================================================
# Directorio de trabajo y librerias
# =============================================================================
globals().clear()
import gc
gc.collect()

import os
#Mateo
#os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
#igal
# os.chdir("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/script/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from scipy.stats import median_abs_deviation , zscore

from Bases_nivel_ncm_12d_6act import *
from procesamiento_nivel_ncm_12d_6act import *
from matriz_nivel_ncm_12d_6act import *
from pre_visualizacion_nivel_ncm_12d_6act import *
from pre_visualizacion_nivel_ncm_12d_6act import *


#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
impo_d12 = pd.read_csv("../data/M_2013a2017_d12.csv", encoding="latin1")      # impo DIARIA
clae = pd.read_csv( "../data/clae_nombre.csv")
comercio = pd.read_csv("../data/comercio_clae.csv", encoding="latin1")
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")
bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018_HS12.csv", sep = ";")
dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")
ncm12_desc = pd.read_csv("../data/d12_2012-2017.csv", sep=";")


#############################################
#           preparación bases               #
#############################################
impo_d12 = predo_impo_all(impo_d12,  name_file ="cuit_explotacion_2013a2017.csv")
ncm12_desc = predo_ncm12_desc(ncm12_desc )["ncm_desc"]
impo_d12  = predo_impo_12d(impo_d12, ncm12_desc)
letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae)
bec_bk = predo_bec_bk(bec)#, bec_to_clae)
dic_stp = predo_stp(dic_stp )

#############################################
#                joins                      #
#############################################
join_impo_clae = def_join_impo_clae(impo_d12, cuit_empresas) #incorpora var destinacion limpia
#join_impo_clae_bec_bk = def_join_impo_clae_bec_bk(join_impo_clae, bec_bk)

# =============================================================================
# EDA BEC5
# =============================================================================
impo_bec = def_join_impo_clae_bec(join_impo_clae, bec)

#join_impo_clae["dest_clean"].value_counts()
# impo_bec[impo_bec["BEC5EndUse"].isnull()]
# impo_d12[impo_d12["descripcion"].isnull()]
bec["BEC5EndUse"].value_counts().sum()
bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]["BEC5EndUse"].value_counts()#.sum()
bec[bec["BEC5EndUse"].str.startswith("INT", na = False)]["BEC5EndUse"].value_counts()#.sum()
bec[bec["BEC5EndUse"].str.startswith("CONS", na = False)]["BEC5EndUse"].value_counts()#.sum()

# =============================================================================
# Bienes de capital
# =============================================================================
# =============================================================================
#  FILTRO STP entre específicos y generales
# =============================================================================
dic_stp["utilizacion"].value_counts()

# vectores filtros
# stp_general = dic_stp[dic_stp["utilizacion"]=="General"] # =~ CI
stp_especifico = dic_stp[dic_stp["utilizacion"].str.contains("Específico|Transporte", case = False)] # =~ BK
join_impo_clae_bec_bk["dest_clean"] = join_impo_clae_bec_bk["destinacion"].apply(lambda x: destinacion_limpio(x))

# opción 1
join_impo_clae_bec_bk["ue_dest"] = np.where(join_impo_clae_bec_bk["HS6"].isin(stp_especifico ["NCM"]), "BK", "")
join_impo_clae_bec_bk["ue_dest"].value_counts()

# filtrado 1
filtro1 = join_impo_clae_bec_bk[join_impo_clae_bec_bk["ue_dest"]==""]

# =============================================================================
# FILTRO DESTINACION
# =============================================================================
filtro1["destinacion"].value_counts()#.sum()
filtro1["dest_clean"].value_counts()#.sum()

# NUEVA PROPUESTA
# partidas_n_dest = join_impo_clae_bec_bk.groupby(["HS6_d12", "destinacion"],as_index = False).size().loc[lambda x: ~x["HS6_d12"].isin(partidas_dest["HS6_d12"])]
ya_filtrado = join_impo_clae_bec_bk[join_impo_clae_bec_bk["ue_dest"]!=""]

(len(filtro1) + len(ya_filtrado)) == len(join_impo_clae_bec_bk)

data_clasif = clasificacion_BK(filtro1)

# =============================================================================
#  Exportacion de datos clasificados con UE dest
# =============================================================================
## DATOS CLASIFICADOS

data_clasif_ue_dest = join_stp_clasif_prop(join_impo_clae_bec_bk, data_clasif)
data_clasif_ue_dest.to_csv("../data/resultados/bk_con_ue_dest.csv")


## DATOS NO CLASIFICADOS
data_not_clasif = pd.concat( [ dest_c], axis = 0)
data_not_clasif["ue_dest"] = "?" #np.NAN

# data_not_clasif.isna().sum()
data_not_clasif["metric"] = data_not_clasif.apply(lambda x: metrica(x), axis = 1)
data_not_clasif["precio_kilo"]= data_not_clasif["valor"]/data_not_clasif["kilos"]

# data_not_clasif.to_csv("../data/resultados/bk_sin_ue_dest.csv")

len(data_not_clasif) + len(data_clasif_ue_dest)


# =============================================================================
# VENN CI
# =============================================================================
cons_int_clasif = clasificacion_CI(impo_bec)
cons_int_clasif ["ue_dest"].value_counts()#.sum()
cons_int_clasif [["filtro", "ue_dest"]].value_counts()#.sum()



# =============================================================================
# VENN CONS
# =============================================================================
cons_fin_clasif = clasificacion_CONS(impo_bec)
cons_fin_clasif [["filtro", "ue_dest"]].value_counts()#.sum()

# =============================================================================
# Consistencia de diagramas
# =============================================================================
# len(impo_bec_ci) + len(impo_bec_bk) + len(impo_bec_cons) == len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )
len(impo_bec_ci) + len(data_not_clasif) + len(data_clasif_ue_dest) + len(impo_bec_cons) == len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )

# impo_ue_dest = pd.concat([pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha", 'metric', 'ue_dest', 'mad', 'median', 'z_score'], axis = 1), bk], axis =0)
cicf_ue_dest = pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha",  'mad', 'median', 'z_score'], axis = 1) #, bk], axis =0)
cicf_ue_dest["precio_kilo"] =  cicf_ue_dest["valor"]/cicf_ue_dest["kilos"]

# bk_ue_dest = pd.read_csv("../data/resultados/bk_con_ue_dest.csv")
bk_ue_dest = data_clasif_ue_dest.copy().drop(['HS4', 'HS4Desc', 'HS6Desc', "BEC5Category"], 1)

# bk_sin_ue_dest = pd.read_csv("../data/resultados/bk_sin_ue_dest.csv")
bk_sin_ue_dest = data_not_clasif.drop(['HS4', 'HS4Desc', 'HS6Desc', "BEC5Category"], 1)

(len(join_impo_clae)-  len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )) - ( len(bk_sin_ue_dest ) + len(bk_ue_dest)+ len(cicf_ue_dest) )

data_model = pd.concat([bk_sin_ue_dest , bk_ue_dest, cicf_ue_dest ], axis = 0) 
data_model ['HS6'] = data_model ['HS6'].astype("str")
data_model ['HS8'] = data_model ['HS6_d12'].str.slice(0,8)
data_model ['HS10'] = data_model ['HS6_d12'].str.slice(0,10)
 
len(join_impo_clae) == (len(data_model) + len(impo_bec[impo_bec["BEC5EndUse"].isnull()] ))

# len(data_model)  - data_model["ue_dest"].value_counts().sum()


# =============================================================================
# Preprocesamiento de Datos para el modelo
# =============================================================================
data_model.info()
data_model["ue_dest"].value_counts()

data_model["actividades"] = data_model["letra1"]+data_model["letra2"]+data_model["letra3"]+data_model["letra4"]+data_model["letra5"]+data_model["letra6"]
data_model["act_ordenadas"] = data_model["actividades"].apply(lambda x: "".join(sorted(x ))) #"".join(sorted(data_model["actividades"]))

data_model.to_csv("../data/resultados/data_modelo_diaria.csv", index = False)

#preprocesamiento etiquetados
cols =  [ "HS6", "HS8", "HS10",   
         'valor',  'kilos', "precio_kilo" , 
         "letra1",	"letra2",	"letra3", 	"letra4", 	"letra5",	"letra6"	,
         "act_ordenadas",
         "uni_est",	"cant_est",	"uni_decl",	"cant_decl",  
         "metric" ,
         "ue_dest"]

data_model = data_model[cols]

#Filtros de columnas
cat_col = list(data_model.select_dtypes(include=['object']).columns)
cat_col.pop(-1)
num_col = list(data_model.select_dtypes(include=['float', "int64" ]).columns)


# data_model = pd.read_csv("../data/resultados/data_modelo_diaria.csv")

data_pre = pd.concat( [ str_a_num(data_model[cat_col]) , data_model[num_col], data_model["ue_dest"] ], axis = 1  )

# datos etiquetados
data_train = data_pre[data_pre ["ue_dest"] != "?" ]
data_train["bk_dummy"] = data_train["ue_dest"].map({"BK": 1, "CI": 0})
data_train.drop("ue_dest", axis = 1, inplace = True)

# datos no etiquetados
data_to_clasif = data_pre[data_pre["ue_dest"] == "?" ]
data_to_clasif.drop("ue_dest", axis = 1, inplace = True)

len(data_pre) == (len(data_train) + len(data_to_clasif))

# exportacion de datos
data_train.to_csv("../data/resultados/data_train_test.csv", index=False)
data_to_clasif .to_csv("../data/resultados/data_to_pred.csv", index=False)



