
import os
# os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
# os.chdir("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/script/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")

import pandas as pd
import numpy as np
from def_bases_nivel_ncm_12d_6act import *
from def_pre_visualizacion_nivel_ncm_12d_6act import *
import datetime


start = datetime.datetime.now()
#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
impo_d12 = pd.read_csv("../data/M_2013a2017_d12.csv", encoding="latin1")      # impo DIARIA

clae = pd.read_csv( "../data/clae_nombre.csv")
dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")
clae_to_ciiu = pd.read_excel("../data/Pasar de CLAE6 a CIIU3.xlsx")

comercio = pd.read_csv("../data/comercio_clae.csv", encoding="latin1")
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")
bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018_HS12.csv", sep = ";")
dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")
ncm12_desc = pd.read_csv("../data/d12_2012-2017.csv", sep=";")


#############################################
#           preparación bases               #
#############################################
# impo_d12 = predo_impo_all(impo_d12,  name_file ="cuit_explotacion_2013a2017.csv")
ncm12_desc = predo_ncm12_desc(ncm12_desc )
impo_d12  = predo_impo_12d(impo_d12, ncm12_desc) # FILTRO AÑO 2017
letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae) #imputacion de actividades faltantes
dic_stp = predo_stp(dic_stp )
dic_propio = predo_dic_propio(clae_to_ciiu, dic_ciiu,clae)

#############################################
#                joins                      #
#############################################
join_impo_clae = def_join_impo_clae(impo_d12, cuit_empresas) #incorpora var destinacion limpia y borra destinacion
join_impo_clae = diccionario_especial(join_impo_clae, dic_propio) #hace el cambio de actividades de CLAE a dicccionario propio
join_impo_clae= def_actividades(join_impo_clae)
impo_bec = def_join_impo_clae_bec(join_impo_clae, bec) #incorpora el cambio de kilogramos : antes habia 177302 uni_decl ==kg y ahora 173036 


# =============================================================================
# EDA BEC5
# =============================================================================
#join_impo_clae["dest_clean"].value_counts()
# impo_bec[impo_bec["BEC5EndUse"].isnull()]
# impo_d12[impo_d12["descripcion"].isnull()]
# bec["BEC5EndUse"].value_counts().sum()
# bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]["BEC5EndUse"].value_counts()#.sum()
# bec[bec["BEC5EndUse"].str.startswith("INT", na = False)]["BEC5EndUse"].value_counts()#.sum()
# bec[bec["BEC5EndUse"].str.startswith("CONS", na = False)]["BEC5EndUse"].value_counts()#.sum()

# =============================================================================
# Bienes de capital
# =============================================================================
#filtro STP
filtro1, filtro_stp, impo_bec_bk = fun_filtro_stp(dic_stp, impo_bec) #filtro1 poseen los BK q se van a seguir procesando

dic_stp["utilizacion"].value_counts()
filtro1["dest_clean"].value_counts()#.sum()

# filtro destinacion
data_clasif_bk , data_not_clasif_bk= clasificacion_BK(filtro1)


# =============================================================================
#  Exportacion de datos clasificados con UE dest
# =============================================================================
## DATOS CLASIFICADOS
data_clasif_ue_dest_bk = join_stp_clasif_prop(impo_bec_bk, data_clasif_bk) #guarda csv
print( "los BK quedan del mismo largo?" , len(data_not_clasif_bk) + len(data_clasif_ue_dest_bk) ==len(impo_bec_bk))

# =============================================================================
# VENN CI
# =============================================================================
cons_int_clasif, impo_bec_ci = clasificacion_CI(impo_bec)
cons_int_clasif["ue_dest"].value_counts()#.sum()
cons_int_clasif [["filtro", "ue_dest"]].value_counts()#.sum()

# =============================================================================
# VENN CONS
# =============================================================================
cons_fin_clasif,impo_bec_cons = clasificacion_CONS(impo_bec)
cons_fin_clasif [["filtro", "ue_dest"]].value_counts()#.sum()

# =============================================================================
# Consistencia de diagramas
# =============================================================================
len(impo_bec_ci) + len(impo_bec_bk) + len(impo_bec_cons) == len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )
len(impo_bec_ci) + len(data_not_clasif_bk) + len(data_clasif_ue_dest_bk) + len(impo_bec_cons) == len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )


# =============================================================================
# Preprocesamiento de Datos para el modelo
# =============================================================================
# Concatenacion de clasificación anterior para el modelo
data_model = concatenacion_ue_dest(cons_fin_clasif, cons_int_clasif,data_clasif_ue_dest_bk,data_not_clasif_bk, join_impo_clae, impo_bec) #concatena y crea algunas vars
data_model["ue_dest"].value_counts()
len(join_impo_clae) == (len(data_model) + len(impo_bec[impo_bec["BEC5EndUse"].isnull()] ))

# preprocesamiento 21 oct
data_pre, data_train, data_to_clasif = predo_datos_modelo_21oct(data_model) #deja los datos listo para entrenar el modelo

# preprocesamiento
# data_pre, data_train, data_to_clasif = predo_datos_modelo(data_model) #deja los datos listo para entrenar el modelo

# # exportacion de datos
# data_train.to_csv("../data/heavys/data_train_test.csv", index=False)
# data_to_clasif.to_csv("../data/resultados/data_to_pred.csv", index=False)
data_model.to_csv("../data/heavys/data_modelo_diaria.csv", index=False)
data_train.to_csv("../data/heavys/data_train_test_21oct.csv", index=False)
data_to_clasif.to_csv("../data/resultados/data_to_pred_21oct.csv", index=False)


end = datetime.datetime.now()
print(end-start)

