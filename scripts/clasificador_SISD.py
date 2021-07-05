# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:54 2021

@author: igalk
"""
import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Bases import *
from procesamiento import *
from matriz import *


# =============================================================================
# Directorio de trabajo
# =============================================================================

#Mateo
os.chdir("C:/Archivos/repos/impo_sectorial/scripts")
os.getcwd()

#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
impo_17 = pd.read_csv(  "data/IMPO_2017.csv", sep=";")
clae = pd.read_csv( "data/clae_nombre.csv")
comercio = pd.read_csv("data/comercio_clae.csv", encoding="latin1")
cuit_clae = pd.read_csv( "data/cuit 2017 impo_con_actividad.csv")
bec = pd.read_csv( "data/HS2012-17-BEC5 -- 08 Nov 2018.csv")
bec_to_clae = pd.read_csv("data/bec_to_clae.csv")

# parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")


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



#############################################
#         ASIGNACIÓN y MATRIZ               #
#############################################

#creamos df para guardar los insumos de la matriz
insumo_matriz = pd.DataFrame()
insumo_matriz ["cuit"]=""
insumo_matriz ["hs6"]=""
insumo_matriz ["valor_pond"]=""    
insumo_matriz ["si"]=""    
insumo_matriz ["sd"]=""
insumo_matriz ["ue_dest"]=""

# matriz_sisd = def_insumo_matriz(insumo_matriz, join_final)
matriz_sisd = pd.read_csv("data/matriz_pesada.csv")

#asignación por probabilidad de G-bk (insumo para la matriz)
matriz_sisd_final = def_matriz_c_prob(matriz_sisd)

#matriz rotada
z = pd.pivot_table(matriz_sisd_final, values='valor_pond', index=['si'], columns=['sd'], aggfunc=np.sum, fill_value=0)
cols=list(z.columns.values)
cols.pop(cols.index("CONS"))
z=z[cols+["CONS"]] #ubicacion del consumo ultima colummna

z= z.append(pd.Series(name='T')) #imputacion de T
z= z.replace(np.nan,0)

#############################################
#             Visualización                 #
#############################################

# transformacion a array de np
z_visual = z.drop(['CONS'], axis=1).to_numpy()

#diagonal y totales col y row
diagonal = np.diag(z_visual)
row_sum = np.nansum(z_visual , axis=1)
col_sum  = np.nansum(z_visual , axis=0)

# sectores
sectores = list(map(chr, range(65, 85)))
sectores_desc = {"A":	"Agricultura y ganadería",  "B":	"Minas y canteras", "C":	"Industria", "D": "Energía",
                 "E":	"Agua y residuos", "F":	"Construcción", "G": "Comercio", "H":	"Transporte",
                 "I":	"Alojamiento", "J":	"Comunicaciones", "K":	"Serv. financieros","L":	"Serv. inmobiliarios",
                 "M":	"Serv. profesionales", "N":	"Serv. apoyo", "O":	"Sector público", "P":	"Enseñanza",
                 "Q":	"Serv. sociales", "R":	"Serv. culturales", "S":	"Serv. personales", "T":	"Serv. doméstico",
                 "U": "Serv. organizaciones"  }
sectores_desc.pop("U")

#diagonal sobre total col y comercio sobre total
diag_total_col = diagonal/col_sum
g_total_col = z_visual[6][:]/col_sum
comercio_y_propio = pd.DataFrame({"Propio": diag_total_col , 'Comercio': g_total_col} , index = sectores_desc.values())

#importaciones totales (ordenadas)
impo_tot_sec = pd.DataFrame({"Importaciones totales": col_sum/(10**6)}, index=sectores_desc.values())  
impo_tot_sec.sort_values(by = "Importaciones totales", ascending= False, inplace = True)

#parametro para graficos
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize': 20,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
 


##### grafico 1
#posiciones para graf
y_pos = np.arange(len(sectores))

plt.bar(y_pos , impo_tot_sec.iloc[:,0])
plt.xticks(y_pos , impo_tot_sec.index, rotation = 45)
plt.title("Importaciones totales destinadas a cada sector", fontsize = 20)
plt.ylabel("Millones de USD")
plt.xlabel("Sector")
plt.savefig('impo_totales.png')


##### grafico 2
#graf division comercio y propio
comercio_y_propio.sort_values(by = 'Propio', ascending = False).plot(kind = "bar", 
                                                                     stacked = True, ylabel = "%", xlabel = "Sector", rot = 45,
                                                                     title = "Sector abastecedor de importaciones (en porcentaje)")
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
# plt.title(fontsize = 15)
plt.savefig('figs/comercio_y_propio.png')


##### grafico 3
# Top 5 de importaciones de cada sector

x = matriz_sisd_final.groupby(["hs6", "sd"], as_index=False)['valor_pond'].sum("valor_pond")
top_5_impo = x.reset_index(drop = True).sort_values(by = ["sd", "valor_pond"],ascending = False)
top_5_impo  = top_5_impo.groupby(["sd"], as_index = True).head(5)
top_5_impo  = pd.merge(left=top_5_impo, right=letras, left_on="sd", right_on="letra", how="left").drop(["sd", "letra"], axis=1)
top_5_impo  = pd.merge(left=top_5_impo, right=bec[["HS6","HS6Desc"]], left_on="hs6", right_on="HS6", how="left").drop("HS6", axis=1)

top_5_impo.to_excel("data/top5_impo.xlsx")











