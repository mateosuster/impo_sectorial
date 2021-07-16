# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:54 2021

@author: igalk
"""
import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from Bases import *
from procesamiento import *
from matriz import *
from pre_visualizacion import *

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
clae = pd.read_csv( "data/clae_agg.csv")
comercio = pd.read_csv("data/comercio_clae.csv", encoding="latin1")
cuit_clae = pd.read_csv( "data/cuit 2017 impo_con_actividad.csv")
bec = pd.read_csv( "data/HS2012-17-BEC5 -- 08 Nov 2018.csv")
bec_to_clae = pd.read_csv("data/bec_to_clae.csv")

# parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")

#carga balance cambiario
bce_cambiario = pd.read_csv("data/balance_cambiario.csv", skiprows = 3, error_bad_lines=False, sep= ";", na_values =['-'])

# carga CIIU
isic = pd.read_csv("data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )
dic_ciiu = pd.read_excel("data/Pasar de CLAE6 a CIIU3.xlsx")
        

#############################################
#           preparación bases               #
#############################################

predo_impo_17(impo_17)

letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)

industria_2d = predo_industria_2d(clae)
comercio_2d = predo_comercio_2d(clae)

cuit_clae_6d = predo_cuit_clae_6d(cuit_clae)

cuit_personas = predo_cuit_clae(cuit_clae_6d, "personas")
cuit_empresas = predo_cuit_clae(cuit_clae_6d, "empresas")
bec_bk = predo_bec_bk(bec, bec_to_clae)

impo_tot_bcra  = predo_bce_cambiario(bce_cambiario )
clae_ciiu = dic_clae_ciiu(isic, dic_ciiu )


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

join_impo_clae_bec_bk_comercio_pond = def_join_impo_clae_bec_bk_comercio_pond(join_impo_clae_bec_bk_comercio)

# join_final = def_calc_pond(join_impo_clae_bec_bk_comercio_pond,tabla_contingencia)
join_final = pd.read_csv("data/resultados/impo_con_ponderaciones_2d.csv",  encoding= 'unicode_escape')


#############################################
#         ASIGNACIÓN y MATRIZ               #
#############################################

# matriz_sisd = def_insumo_matriz( join_final)
# matriz_sisd.to_csv("data/matriz_pesada_2d.csv", index = False) 
matriz_sisd = pd.read_csv("data/resultados/matriz_pesada_2d.csv").drop("Unnamed: 0", axis = 1)

#asignación por probabilidad de G-bk (insumo para la matriz)
matriz_sisd_final = def_matriz_c_prob(matriz_sisd)

#matriz rotada
z = pd.pivot_table(matriz_sisd_final, values='valor_pond', index=['si'], columns=['sd'], aggfunc=np.sum, fill_value=0)

#busco las filas que faltan
# set(list(z.columns)).symmetric_difference( set(list( z.index ) ) )
# [x for x in list(z.index)  if x not in  list(z.columns) ] # OK pero son tipos de datos distintos


#############################################
#             Visualización                 #
#############################################

# parametro para graficos
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize': 30,
         'xtick.labelsize':12,
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
 

##### grafico 1
impo_tot_sec = impo_total(z, industria_2d)

#posiciones para graf
y_pos = np.arange(len(industria_2d["desc"]))

plt.bar(y_pos , impo_tot_sec.iloc[:,0])
plt.xticks(y_pos , impo_tot_sec.index, rotation = 75)
plt.title("Importaciones de bienes de capital destinadas a industria manufacturera", fontsize = 30)
plt.ylabel("Millones de USD")
plt.xlabel("Sector \n \n Fuente: CEPXXI en base a Aduana y AFIP")
plt.tight_layout()
plt.savefig('data/resultados/impo_totales_2d.png')
    

##### grafico 2
#graf division comercio y propio
comercio_y_propio  = impo_comercio_y_propio(z, industria_2d , comercio_2d)
ax = comercio_y_propio.plot(kind = "bar",  rot = 75, stacked = True, ylabel = "%", xlabel = "Sector \n \n Fuente: CEPXXI en base a Aduana y AFIP")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
plt.title("Sector abastecedor de importaciones de bienes de capital de la industria (en porcentaje)", fontsize = 30)
plt.tight_layout(pad=3)
plt.savefig('data/resultados/comercio_y_propio_2d.png')


##### tabla 1 
# Top 5 de importaciones de cada sector
top_5_impo = top_5(matriz_sisd_final, industria_2d, bec, impo_tot_sec)
top_5_impo.to_excel("data/resultados/top_5_impo_2d.xlsx")


#### grafico 3
comparacion_bcra = merge_bcra(impo_tot_sec, impo_tot_bcra)
comparacion_bcra.plot(x="sector", y = ["impo_sisd", "impo_bcra"], kind="bar", rot=15,
                 ylabel = "Millones de dólares", xlabel = "Sector \n \n Fuente: CEPXXI en base a BCRA, Aduana, AFIP y UN Comtrade. La estimación del BCRA no es exclusiva de Bienes de Capital")#,)
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(pad=7)
plt.title( "Importacion sectorial de bienes de capital de la industria manufacturera \n Comparación de estimaciones",  fontsize = 30)
plt.legend(["SI-SD (BK)", "BCRA (BK+CI+CONS)"])
plt.savefig("data/resultados/comparacion_bcra_2d.png")



#### grafico 4
comparacion_ciiu = join_sisd_ciiu(join_impo_clae_bec_bk, clae_ciiu ,impo_tot_sec,  industria_2d)
comparacion_ciiu.plot(x="desc", y = ["impo_sisd", "impo_ciiu"], kind="bar", rot=15, color = ["C1", "g"],
                 ylabel = "Millones de dólares", xlabel = "Sector \n \n Fuente: CEPXXI en base a Aduana, AFIP y UN Comtrade")
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(pad=7)
plt.title( "Importacion sectorial de bienes de capital de la industria manufacturera \n Comparación de estimaciones",  fontsize = 30)
plt.legend(["SI-SD (BK)", "CIIU (BK)"])
plt.savefig("data/resultados/comparacion_ciiu_2d.png")


# Grafico 5
#data
tidy_bcra = pd.melt(comparacion_bcra.rename(columns= {"impo_sisd": "SISD", "impo_bcra":"BCRA"}), id_vars = "sector", value_vars = ["SISD", "BCRA"])
tidy_ciiu = pd.melt(comparacion_ciiu.rename(columns= {"impo_sisd": "SISD", "impo_ciiu":"CIIU"}), id_vars = "desc", value_vars = ["SISD", "CIIU"])

sns.set_context('talk')
sns.set(font_scale = 2)
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(26,15))
plt.suptitle("Importacion de bienes de capital de la industria manufacturera", size = 30)

sns.barplot(x = "sector", y = "value", hue ="variable",ax=ax[0] , data = tidy_bcra, palette = ["C1", "b"])#, hue_order= ['SISD', 'BCRA'])
ax[0].set_title("Comparación con estimación BCRA",  fontsize = 20)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 35, size = 17)
ax[0].set(xlabel= "", ylabel= "Millones de dólares")
ax[0].legend(title='')


sns.barplot(data = tidy_ciiu , x = "desc" , y = "value", hue = "variable", ax=ax[1], palette = ["C1", "g"])
ax[1].set_title('Comparación con estimación CIIU',  fontsize = 20)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 35, size = 17)
ax[1].set(xlabel= "", ylabel= "Millones de dólares")
sns.set_style("ticks", {"xtick.major.size":1, "ytick.major.size":8})
ax[1].legend(title='')#, labels=['SISD', 'BCRA'])
fig.tight_layout()

plt.savefig("data/resultados/comparacion_2d.png")
