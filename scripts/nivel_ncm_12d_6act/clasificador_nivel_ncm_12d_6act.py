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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
# import plotinpy as pnp
import re
import tqdm

import datatable as dt

# import prince
# from prince import ca
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

#STP
# dic_stp = pd.read_excel("C:/Users/igalk/OneDrive/Documentos/laburo/CEP/procesamiento impo/nuevo1/impo_sectorial//scripts/data/bsk-prod-clasificacion.xlsx")

# ISIC y CLAE
hs_to_isic = pd.read_csv("../data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )
dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")
clae_to_ciiu = pd.read_excel("../data/Pasar de CLAE6 a CIIU3.xlsx")


#############################################
#           preparación bases               #
#############################################

# predo_impo_17(impo_17)
ncm12_desc = predo_ncm12_desc(ncm12_desc )["ncm_desc"]    
impo_d12  = predo_impo_12d(impo_d12, ncm12_desc)
letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae)
bec_bk = predo_bec_bk(bec)#, bec_to_clae)
dic_stp = predo_stp(dic_stp)
datos = predo_datamodel(data_predichos)



#############################################
#                joins:                     ESTA PARTE ES INNCESARIA #
#############################################
join_impo_clae = def_join_impo_clae(impo_d12, cuit_empresas) #join CUIT
join_impo_clae_bec_bk = def_join_impo_clae_bec(join_impo_clae, bec_bk) # filtro BK

############################################################
#  Asignación por STP / modificación de actividades x ncm  #
############################################################

#revisión de nulos de BEC5EndUSE
impo_bec = pd.merge(join_impo_clae, bec[["HS6", "BEC5EndUse" ]], how= "left" , left_on = "HS6", right_on= "HS6" )
(len(datos ) + len(impo_bec[impo_bec["BEC5EndUse"].isnull()]) ) == len(join_impo_clae)

datos_bk = asignacion_stp_BK(datos, dic_stp)
join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(datos_bk , comercio)


datos_bk.info()

clae_to_ciiu.info()
clae_to_ciiu[clae_to_ciiu["ciiu3_4c"].isnull() ]
clae_to_ciiu.dropna(inplace = True)
clae_to_ciiu["ciiu3_4c"] = clae_to_ciiu["ciiu3_4c"].astype(int)
clae_to_ciiu[clae_to_ciiu.duplicated()] 

datos_bk_a_ciiu= datos_bk[["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6"]].copy()

for clae, ciiu_name in zip(["actividad1", "actividad2", "actividad3", "actividad4", "actividad5", "actividad6"],
                           ["ciiu_act1", "ciiu_act2", "ciiu_act3", "ciiu_act4", "ciiu_act5", "ciiu_act6"]):
    # print(clae, ciiu_name)
    ciiu_data = clae_to_ciiu.copy()
    ciiu_data = clae_to_ciiu[["clae6", "ciiu3_4c"]]
    ciiu_data.rename(columns = {"ciiu3_4c": ciiu_name }, inplace=True)
    datos_bk_a_ciiu= pd.merge(datos_mergeados , ciiu_data, how = "left" ,left_on = clae, right_on= "clae6" ).drop("clae6", 1)

datos_mergeados = pd.merge(datos_bk , clae_to_ciiu, how = "left" ,left_on = "actividad1", right_on= "clae6" )    
datos_mergeados.head()

clae1 = pd.DataFrame(datos_bk["actividad1"].unique(), columns = ["act1"])
ciiu1 =clae_to_ciiu.copy()#[["clae6", "ciiu3_4c"]]
datos_mergeados = pd.merge(clae1 , ciiu1, how = "left" ,left_on = "act1", right_on= "clae6" )    
x = datos_mergeados[datos_mergeados.duplicated("act1")]
ciiu1[ciiu1.duplicated("clae")]

####################
# obtencion de LETRA_nn
dic = []
dic_list = []

letra1 = datos_bk.columns.get_loc("letra1") + 1
letra2 = datos_bk.columns.get_loc("letra2") + 1
letra3 = datos_bk.columns.get_loc("letra3") + 1
letra4 = datos_bk.columns.get_loc("letra4") + 1
letra5 = datos_bk.columns.get_loc("letra5") + 1
letra6 = datos_bk.columns.get_loc("letra6") + 1

actividad1 = datos_bk.columns.get_loc("actividad1") + 1
actividad2 = datos_bk.columns.get_loc("actividad2") + 1
actividad3 = datos_bk.columns.get_loc("actividad3") + 1
actividad4 = datos_bk.columns.get_loc("actividad4") + 1
actividad5 = datos_bk.columns.get_loc("actividad5") + 1
actividad6 = datos_bk.columns.get_loc("actividad6") + 1


for x in datos_bk.itertuples():
    dic = []
    for letra, act, letra_name in zip([letra1, letra2, letra3,letra4, letra5, letra6],
                                 [actividad1, actividad2, actividad3,actividad4, actividad5, actividad6],
                                 ["letra1", "letra2", "letra3","letra4", "letra5", "letra6"]):
        
        if x[letra] in ["D", "H", "I"]:
            new_letra= x[letra] + str(x[act])[0:2]
        else:
            new_letra= x[letra] 
            
        # dic[letra_name] = new_letra
        dic.append(new_letra)
        
    dic_list.append(dic)    
    
x= pd.DataFrame.from_dict(dic_list)    
    
datos_bk["actividad1"].astype(str).str.slice(0,2)

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
# join_final.to_csv("../data/resultados/impo_con_ponderaciones_12d_6act_post_ml.csv", index=False)
join_final = pd.read_csv("../data/resultados/impo_con_ponderaciones_12d_6act_post_ml.csv")

# selecciono columnas del join final
# filtro = ["HS6", "CUIT_IMPOR", "valor", "letra1", "letra2", "letra3", 
# "vta_bk", "vta_sec", "vta_bk2", "vta_sec2", "vta_bk3", "vta_sec3", 
# "letra1_pond", "letra2_pond", "letra3_pond"]
# join_final = join_final.sort_values("HS6")[filtro]

#############################################
#         ASIGNACIÓN y MATRIZ               #
#############################################

# matriz_sisd = def_insumo_matriz(join_final)
# matriz_sisd.to_csv("../data/resultados/matriz_pesada_12d_6act_postML.csv", index= False)
matriz_sisd = pd.read_csv("../data/resultados/matriz_pesada_12d_6act_postML.csv")

#asignación por probabilidad de G-bk (insumo para la matriz)
matriz_sisd_final = def_matriz_c_prob(matriz_sisd)

#matriz rotada
z =to_matriz(matriz_sisd_final)
z_si_sd = pd.pivot_table(matriz_sisd_final, values='valor_pond', index=['hs6_d12'], columns=['sd'], aggfunc=np.sum, fill_value=0) 


#############################################
#             Matriz Numpy                 #
#############################################

sectores_desc = sectores()
impo_tot_sec = impo_total(z, sectores_desc)
comercio_y_propio = impo_comercio_y_propio(z, sectores_desc)


# =============================================================================
# CIIU
# =============================================================================
impo_ciiu_letra = impo_ciiu_letra(isic, dic_ciiu,datos_bk )








# =============================================================================
#                       Visualizacion
# =============================================================================
#parametro para graficos
params = {'legend.fontsize': 20,
          'figure.figsize': (20, 10),
         'axes.labelsize': 15,
         'axes.titlesize': 30,
          'xtick.labelsize':20,
          'ytick.labelsize':20
         }
plt.rcParams.update(params)
 

##### grafico 1
#posiciones para graf
y_pos = np.arange(len(sectores_desc.values())) 

plt.bar(y_pos , impo_tot_sec.iloc[:,0]/(10**6) )
plt.xticks(y_pos , impo_tot_sec.index, rotation = 75)
plt.title("Importaciones de bienes de capital destinadas a cada sector")#, fontsize = 30)
plt.ylabel("Millones de USD")
plt.xlabel("Sector \n \n Fuente: CEPXXI en base a Aduana, AFIP y UN Comtrade")
# plt.subplots_adjust(bottom=0.7,top=0.83)
plt.tight_layout()
plt.savefig('data/resultados/impo_totales_letra.png')


##### grafico 2
#graf division comercio y propio
ax = comercio_y_propio.plot(kind = "bar", rot = 75,
                            stacked = True, ylabel = "%", 
                            xlabel = "Sector \n \n Fuente: CEPXXII en base a Aduana, AFIP y UN Comtrade")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(pad=3)
plt.title( "Sector abastecedor de importaciones de bienes de capital (en porcentaje)")#,  fontsize = 30)

plt.savefig('data/resultados/comercio_y_propio_letra.png')


##### insumos tabla 1
# Top 5 de importaciones de cada sector
top_5_impo = top_5(matriz_sisd_final, letras, bec, impo_tot_sec)

#CUITS QUE IMPORTAN TOP HS6 Industria
# top_industria = top_5_impo[top_5_impo["letra"]=="C"]["hs6"].iloc[[0,3]]
# cuit_top_c = join_impo_clae[join_impo_clae["HS6"].isin(top_industria )].sort_values("valor",ascending=False)#["CUIT_IMPOR"].unique()
# cuit_empresas[cuit_empresas["cuit"].isin(cuit_top_c)]

# top_5_impo.to_csv("../data/resultados/top5_impo.csv")
top_5_impo.to_excel("../data/resultados/top5_impo.xlsx")


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



