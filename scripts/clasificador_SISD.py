# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:54 2021

@author: igalk
"""
import os 
os.getcwd()

import pandas as pd
import numpy as np
import matplotlib as math


from Bases import *
from procesamiento import *

#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
# impo_17 = pd.read_csv(  "C:/Archivos/repos/impo_sectorial/scripts/data/IMPO_2017.csv", sep=";")
# clae = pd.read_csv( "C:/Archivos/repos/impo_sectorial/scripts/data/clae_nombre.csv")
# comercio = pd.read_csv("C:/Archivos/repos/impo_sectorial/scripts/data/comercio_clae.csv", encoding="latin1")
# cuit_clae = pd.read_csv( "C:/Archivos/repos/impo_sectorial/scripts/data/cuit 2017 impo_con_actividad.csv")
# bec = pd.read_csv( "C:/Archivos/repos/impo_sectorial/scripts/data/HS2012-17-BEC5 -- 08 Nov 2018.csv")
# #parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# #transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")
# bec_to_clae = pd.read_csv("C:/Archivos/repos/impo_sectorial/scripts/data/bec_to_clae.csv")



impo_17 = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/IMPO_2017.csv", sep=";")
clae = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/clae_nombre.csv")
comercio = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/comercio_clae.csv", encoding="latin1")
cuit_clae = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/cuit 2017 impo_con_actividad.csv")
bec = pd.read_csv( "C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/HS2012-17-BEC5 -- 08 Nov 2018.csv")
#parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
#transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")
bec_to_clae = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/bec_to_clae.csv")



#############################################
#           preparación bases               #
#############################################

predo_impo_17(impo_17)
letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)
cuit_personas = predo_cuit_clae(cuit_clae, "personas")
cuit_empresas = predo_cuit_clae(cuit_clae, "empresas")
bec_bk = predo_bec_bk(bec, bec_to_clae)


#############################################
#                joins                      #
#############################################

join_impo_clae = def_join_impo_clae(impo_17, cuit_empresas)
join_impo_clae_bec_bk = def_join_impo_clae_bec(join_impo_clae, bec_bk)
join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio)



#############################################
#           Tabla de contingencia           #
#              producto-sector              #
#############################################

tabla_contingencia = def_contingencia(join_impo_clae_bec_bk_comercio)

#############################################
#      ponderación por ncm y letra          #
#############################################

join_impo_clae_bec_bk_comercio_pond = def_join_impo_clae_bec_bk_comercio_pond(join_impo_clae_bec_bk_comercio, tabla_contingencia)

join_final = def_calc_pond(join_impo_clae_bec_bk_comercio_pond,tabla_contingencia)
print (join_final)


#############################################
#         ASIGNACIÓN y MATRIZ               #
#############################################






#############################################
#             Visualización                 #
#############################################
















