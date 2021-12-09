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
#filtro STP
filtro1, impo_bec_bk = filtro_stp(dic_stp, impo_bec)
ya_filtrado = impo_bec_bk[impo_bec_bk["ue_dest"]!=""]

dic_stp["utilizacion"].value_counts()
filtro1["destinacion"].value_counts()#.sum()
filtro1["dest_clean"].value_counts()#.sum()

(len(filtro1) + len(ya_filtrado)) == len(impo_bec_bk)

# filtro destinacion
data_clasif , data_not_clasif= clasificacion_BK(filtro1)

# =============================================================================
#  Exportacion de datos clasificados con UE dest
# =============================================================================
## DATOS CLASIFICADOS
data_clasif_ue_dest = join_stp_clasif_prop(impo_bec_bk, data_clasif)
data_clasif_ue_dest.to_csv("../data/resultados/bk_con_ue_dest.csv")

len(data_not_clasif) + len(data_clasif_ue_dest) ==len(impo_bec_bk)

# =============================================================================
# VENN CI
# =============================================================================
cons_int_clasif, impo_bec_ci = clasificacion_CI(impo_bec)
cons_int_clasif ["ue_dest"].value_counts()#.sum()
cons_int_clasif [["filtro", "ue_dest"]].value_counts()#.sum()



# =============================================================================
# VENN CONS
# =============================================================================
cons_fin_clasif,impo_bec_cons = clasificacion_CONS(impo_bec)
cons_fin_clasif [["filtro", "ue_dest"]].value_counts()#.sum()

# =============================================================================
# Consistencia de diagramas
# =============================================================================
# len(impo_bec_ci) + len(impo_bec_bk) + len(impo_bec_cons) == len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )
len(impo_bec_ci) + len(data_not_clasif) + len(data_clasif_ue_dest) + len(impo_bec_cons) == len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )

# datos para el modelo
data_model = datos_modelo(cons_fin_clasif, cons_int_clasif,data_clasif_ue_dest,data_not_clasif)


len(join_impo_clae) == (len(data_model) + len(impo_bec[impo_bec["BEC5EndUse"].isnull()] ))
len(data_model)  - data_model["ue_dest"].value_counts().sum()

# =============================================================================
# Preprocesamiento de Datos para el modelo
# =============================================================================
data_model.info()
data_model["ue_dest"].value_counts()


def str_a_num(df):
  for  (columnName, columnData)  in df.iteritems():
  # for col in df:
    # original = np.sort(df[col].unique())
    original = np.sort(columnData.values.unique())
    reemplazo = range(len(original))
    mapa = dict(zip(original, reemplazo))
    df.loc[:,col] = df.loc[:,col].replace(mapa)
  return(df)

def predo_datos_modelo(data_model):
# data_model = pd.read_csv("../data/resultados/data_modelo_diaria.csv")
# Filtros de columnas
    cat_col = list(data_model.select_dtypes(include=['object']).columns)
    cat_col.pop(-1)
    num_col = list(data_model.select_dtypes(include=['float', "int64"]).columns)

    data_pre = pd.concat( [ str_a_num(data_model[cat_col]) , data_model[num_col], data_model["ue_dest"] ], axis = 1  )

    # datos etiquetados
    data_train = data_pre[data_pre ["ue_dest"] != "?" ]
    data_train["bk_dummy"] = data_train["ue_dest"].map({"BK": 1, "CI": 0})
    data_train.drop("ue_dest", axis = 1, inplace = True)

    # datos no etiquetados
    data_to_clasif = data_pre[data_pre["ue_dest"] == "?" ]
    data_to_clasif.drop("ue_dest", axis = 1, inplace = True)

    print(len(data_pre) == (len(data_train) + len(data_to_clasif)))

    return data_pre, data_train, data_to_clasif

data_pre, data_train, data_to_clasif = predo_datos_modelo(data_model)

# exportacion de datos
data_train.to_csv("../data/resultados/data_train_test.csv", index=False)
data_to_clasif .to_csv("../data/resultados/data_to_pred.csv", index=False)



