# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:48:20 2021

@author: mateo
"""

import os 
#Mateo
# os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")
#igal
# os.chdir("C:/Users/igalk/OneDrive/Documentos/laburo/CEP/procesamiento impo/nuevo1/impo_sectorial/scripts/nivel_ncm_12d_6act")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import plotinpy as pnp

# import prince
# from prince import ca
from def_bases_nivel_ncm_12d_6act import *
from def_procesamiento_nivel_ncm_12d_6act import *
from def_matriz_nivel_ncm_12d_6act import *
from def_pre_visualizacion_nivel_ncm_12d_6act import *


#############################
            # DATOS
#############################

dic_propio = pd.read_csv("../data/resultados/dic_clae_ciiu_propio.csv")
ncm12_desc = pd.read_csv("../data/d12_2012-2017.csv", sep=";")
ncm12_desc = predo_ncm12_desc(ncm12_desc )   
cuits_desc = pd.read_csv("../data/resultados/cuits_unicos.csv")

dic_propio[["propio_letra_2", "desc"]].drop_duplicates().to_csv("../data/resultados/desc_actividades.csv", index=False, sep=";", encoding='utf-8-sig')

# BK
matriz_sisd_bk = pd.read_csv("../data/resultados/matriz_sisd.csv").set_index("si")
asign_pre_matriz= pd.read_csv("../data/resultados/asign_pre_matriz.csv")

# CI
asign_pre_matriz_ci = pd.read_csv("../data/resultados/asign_pre_matriz_ci.csv")
matriz_sisd_ci = pd.read_csv("../data/resultados/matriz_sisd_ci.csv")


##########################
# Resultados propios 
##########################


# =============================================================================
# Impo total 
# =============================================================================
impo_tot_sec, comercio_y_propio = impo_total(matriz_sisd_bk, dic_propio, sectores_desc= False, largo_actividad=30)
impo_tot_sec_ci, comercio_y_propio_ci  = impo_total(matriz_sisd_ci,dic_propio, sectores_desc= False, largo_actividad=30)

# graficos
graficos(dic_propio, impo_tot_sec, comercio_y_propio, ue_dest = "bk", largo_actividad=30)
graficos(dic_propio, impo_tot_sec_ci, comercio_y_propio_ci,  ue_dest= "ci", largo_actividad=30)

# =============================================================================
#                   Top n de importaciones de cada sector
# =============================================================================
top_5_impo = top_5(asign_pre_matriz, ncm12_desc, impo_tot_sec,dic_propio, bien = "bk", n=10) 
top_5_impo_ci = top_5(asign_pre_matriz_ci, ncm12_desc,  impo_tot_sec_ci, dic_propio, bien = "ci", n=10)

# =============================================================================
#  top 50 de productos, los primeros 5 sectores importadores
# =============================================================================
top_productos = def_top_hs(asign_pre_matriz, ncm12_desc, "bk")
top_productos_ci = def_top_hs(asign_pre_matriz_ci, ncm12_desc, "ci")

top_sd_de_top_hs = def_top_sd_de_top_hs(asign_pre_matriz, ncm12_desc, dic_propio, top_productos, "bk")
top_sd_de_top_hs_ci = def_top_sd_de_top_hs(asign_pre_matriz_ci, ncm12_desc, dic_propio, top_productos_ci, "ci")


# =============================================================================
# 10 primeros cuit (razón social) importadores por sector
# =============================================================================
top_cuits = def_top_cuits(asign_pre_matriz, dic_propio, "bk", cuits_desc )
top_cuits_ci = def_top_cuits(asign_pre_matriz_ci, dic_propio, "ci", cuits_desc )


# =============================================================================
# 10 primeros cuit importadores de los 50 productos más importados
# =============================================================================
top_cuit_de_top_hs = def_top_cuit_de_top_hs(asign_pre_matriz, ncm12_desc, dic_propio, top_productos, "bk", cuits_desc)
top_cuit_de_top_hs = def_top_cuit_de_top_hs(asign_pre_matriz_ci, ncm12_desc, dic_propio, top_productos, "ci", cuits_desc)


# =============================================================================
#  Tree map
# =============================================================================
top_5_impo


# =============================================================================
# # VIEJO
# =============================================================================
######################
#CUITS QUE IMPORTAN TOP HS6 Industria
# top_industria = top_5_impo[top_5_impo["letra"]=="C"]["hs6"].iloc[[0,3]]
# cuit_top_c = join_impo_clae[join_impo_clae["HS6"].isin(top_industria )].sort_values("valor",ascending=False)#["CUIT_IMPOR"].unique()
# cuit_empresas[cuit_empresas["cuit"].isin(cuit_top_c)]


# =============================================================================
# #COMPARACIONES CON OTRAS ESTIMACIONES
# =============================================================================

# CIIU
impo_ciiu_letra = impo_ciiu_letra(hs_to_isic , dic_ciiu, datos_bk )

##### grafico 3
########### STP
stp = pd.read_csv("../data/bsk-prod-series.csv")
stp["anio"] = stp["indice_tiempo"].str.slice(0,4)
stp_anual = stp.groupby("anio")["transporte"].sum()

transporte_stp = stp_anual.loc["2017"]
    
######### COMPARACION CON BCRA Y CIIU
impo_bcra_letra=predo_bce_cambiario(bce_cambiario)
impo_ciiu_letra = impo_ciiu_letra(isic, dic_ciiu,join_impo_clae_bec_bk )

comparacion = pd.merge(impo_tot_sec, impo_bcra_letra, left_on= "letra", right_on= "letra", how="right" )
comparacion = pd.merge(comparacion, impo_ciiu_letra, left_on= "letra", right_on= "letra", how="left" )

comparacion = pd.merge(comparacion, pd.DataFrame( {"letra": sectores_desc.keys(), "desc":sectores_desc.values() }), left_on= "letra", right_on= "letra", how="left" )
comparacion["impo_tot"] = comparacion["impo_tot"] /(10**6)
comparacion["impo_bcra"] = comparacion["impo_bcra"]
comparacion["impo_ciiu"] = comparacion["impo_ciiu"] 



comparacion.sort_values(by = 'impo_tot', ascending = False).plot(x="desc", y = ["impo_tot", "impo_bcra"
                                                                                # ,"impo_ciiu"
                                                                                ], kind="bar", rot=75,
                 ylabel = "Millones de dólares", xlabel = "Sector \n \n Fuente: CEPXXI en base a BCRA, Aduana, AFIP y UN Comtrade. La estimación del BCRA no es exclusiva de Bienes de Capital")#,)
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(pad=3)
plt.title( "Importacion sectorial. Comparación de estimaciones")
plt.legend(["SI-SD (BK)", "BCRA (BK+CI+CONS)", "CIIU (BK)"])
# pnp.plot_bars_with_breaks(
#      [1, 20, 30],
#     [(15, 25)]
#     )

plt.savefig('../data/resultados/comparacion_estimaciones.png')

comparacion_tidy = comparacion.melt(id_vars= "desc", value_vars=["impo_tot", "impo_bcra", "impo_ciiu"])#,var_name ="value" )
sns.barplot(data=comparacion_tidy , x="desc", y ="value", hue="variable")


# =============================================================================
# Análisis de correspondencia
# =============================================================================

matriz_mca= predo_mca(matriz_sisd_final, "count") #"np.sum"

ca = prince.CA()
ca = ca.fit(matriz_mca)

#ca.plot_coordinates(matriz_mca,show_col_labels=False, show_row_labels=True, figsize=(8, 8))
x = ca.row_coordinates(matriz_mca)
y = ca.column_coordinates(matriz_mca)

#sectores al autovalor de x
x["sector"] = matriz_mca.index
z=["A","B","C","D","E","F"]
j=["CONS"]
x["tipo_sector"] = np.where(x['sector'].isin(z), 'Bs', 'Ss')
x["tipo_sector"] = np.where(x['sector'].isin(j), 'CONS', x["tipo_sector"])
#matriz_mca = matriz_mca.drop(["sector"], axis=1)
#print (x)


#labels
labels = ca.explained_inertia_
x_label = 'Component 0 ('+ str(round(labels[0]*100,2))+'% inertia)'
y_label = 'Component 1 ('+ str(round(labels[1]*100,2))+'% inertia)'

#figure

colors = {'Bs':'red', 'Ss':'darkmagenta', 'CONS':'yellow'}

plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='k')
ax2 = plt.scatter(y[0], y[1], marker='x', label='NCM')
ax = plt.scatter(x[0], x[1], marker='^', label='Sectores', c=x['tipo_sector'].map(colors))
# ax = plt.scatter(x[0], x[1], marker=x["sector"], label='Sectores', c=x['tipo_sector'].map(colors))

# q = x[0]
# p = x[1]
# for xi, yi, label in zip(q, p, x["sector"] ):
#     plt.scatter.annotate(label, (xi, yi))

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid()
plt.legend()
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.title("Análisis de correspondencia con suma de total importado")






