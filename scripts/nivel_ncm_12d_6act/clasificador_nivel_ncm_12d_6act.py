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
# os.chdir("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/script/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.getcwd()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotinpy as pnp


# import prince
# from prince import ca


from Bases_nivel_ncm_12d_6act import *
from procesamiento_nivel_ncm_12d_6act import *
from matriz_nivel_ncm_12d_6act import *
from pre_visualizacion_nivel_ncm_12d_6act import *


#############################################
# Cargar bases con las que vamos a trabajar #
#############################################

#impo 12 d
impo_d12 = pd.read_csv("../data/IMPO_2017_12d.csv")
#impo_17 = pd.read_csv(  "../data/IMPO_2017.csv", sep=";")

clae = pd.read_csv( "../data/clae_nombre.csv")
comercio = pd.read_csv("../data/comercio_clae.csv", encoding="latin1")

#cuit_clae = pd.read_csv( "../data/cuit 2017 impo_con_actividad.csv")
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")

bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018.csv")
bec_to_clae = pd.read_csv("../data/bec_to_clae.csv")

#diccionario ncm12d
ncm12_desc = pd.read_csv("../data/NCM 12d.csv", sep=";")
ncm12_desc_split = pd.concat([ncm12_desc.iloc[:,0], pd.DataFrame(ncm12_desc['Descripción Completa'].str.split('//', expand=True))], axis=1)

# parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")

# bce_cambiario = pd.read_csv("../data/balance_cambiario.csv", skiprows = 3, error_bad_lines=False, sep= ";", na_values =['-'])
# isic = pd.read_csv("../data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )
# dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")

#STP
# dic_stp = pd.read_excel("C:/Archivos/repos/impo_sectorial/scripts/data/bsk-prod-clasificacion.xlsx")


#############################################
#           preparación bases               #
#############################################

# predo_impo_17(impo_17)
impo_d12  = predo_impo_12d(impo_d12, ncm12_desc)

letras = predo_sectores_nombres(clae)

comercio = predo_comercio(comercio, clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae)

bec_bk = predo_bec_bk(bec, bec_to_clae)
#dic_stp = predo_stp(dic_stp )




#############################################
#                joins                      #
#############################################


join_impo_clae = def_join_impo_clae(impo_d12, cuit_empresas)
join_impo_clae_bec_bk = def_join_impo_clae_bec(join_impo_clae, bec_bk)
join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio)


# =============================================================================
# Implementacion STP para agro y transporte
# =============================================================================

dic_stp[dic_stp["desc"].str.contains("cami", case =False)]
dic_stp[dic_stp["desc"].str.contains("cose", case =False)]
dic_stp[dic_stp["NCM"]==870120]

stp_transporte = dic_stp[dic_stp["demanda"].str.contains("trans", case =False)]
stp_agro = dic_stp[dic_stp["demanda"].str.contains("agríc", case =False)]


#############################################
#           Tabla de contingencia           #
#              producto-sector              #
#############################################

tabla_contingencia = def_contingencia(join_impo_clae_bec_bk_comercio)

#############################################
#      ponderación por ncm y letra          #
#############################################

join_impo_clae_bec_bk_comercio_pond = def_join_impo_clae_bec_bk_comercio_pond(join_impo_clae_bec_bk_comercio, tabla_contingencia)

#join_final = def_calc_pond(join_impo_clae_bec_bk_comercio_pond,tabla_contingencia)
#join_final.to_csv("../data/resultados/impo_con_ponderaciones_12d_6act.csv", index=False)
#join_final = pd.read_csv("../data/resultados/impo_con_ponderaciones_12d_6act.csv")

# filtro = ["HS6", "CUIT_IMPOR", "valor", "letra1", "letra2", "letra3", 
# "vta_bk", "vta_sec", "vta_bk2", "vta_sec2", "vta_bk3", "vta_sec3", 
# "letra1_pond", "letra2_pond", "letra3_pond"]
# join_final = join_final.sort_values("HS6")[filtro]


# =============================================================================
# Pruebas
# =============================================================================
ncm_sector = pd.read_csv("../nivel_clae_letra/resultados/asignacion_ncm.csv")


# =============================================================================
# construccion de  features
# =============================================================================

#cantidad de empresas que importan la partida y cantidad y valor importada total x partida
indicadores_raw= join_impo_clae_bec_bk.groupby(["cuit", "unidad_medida", "HS6_d12"], as_index = False).agg({ "cantidad":"sum", "valor": "sum"})
indicadores_raw["precio_empresa" ] = indicadores_raw.apply( lambda x: x.valor / x.cantidad, axis = 1 )

precio_emp_avg = indicadores_raw.groupby( ["unidad_medida", "HS6_d12"], as_index= False)["precio_empresa"].mean().rename(columns = {"precio_empresa": "precio_emp_avg"})

indicadores = indicadores_raw.reset_index().groupby([ "unidad_medida", "HS6_d12"], as_index = False).agg({"cuit": "count", "cantidad":"sum", "valor": "sum"})
indicadores .rename(columns= {"cuit": "n_emp_importadoras", 
                            "cantidad": "cant_importada",
                            "valor": "valor_total"}, inplace = True)
# Productos por empresa
indicadores["cant_x_empresa"] = indicadores["cant_importada"]/indicadores["n_emp_importadoras"]

#precio
indicadores["precio_sin_ponderar"] = indicadores["valor_total"]/indicadores["cant_importada"]

indicadores = pd.merge(indicadores, precio_emp_avg.drop("unidad_medida", axis =1) , how = "left", left_on = "HS6_d12",right_on = "HS6_d12")

indicadores["control_precio"] = indicadores["precio_sin_ponderar"]/indicadores["precio_emp_avg"] -1 

#PRECIOS
# Precio unitario (todos los productos)
join_impo_clae_bec_bk["precio"] =join_impo_clae_bec_bk["valor"]/join_impo_clae_bec_bk["cantidad"]  

## sesgos de los precios
sesgos = join_impo_clae_bec_bk.groupby("HS6_d12", as_index=False)["precio"].skew(axis=0, skipna=False).rename(columns = {"precio": "asimietria_precio"})
sesgos["asimietria_precio"].hist(bins = 100) 
plt.title("Histogramas de las asimetrias de los precios unitarios")

sesgos_neg  = sesgos[sesgos<0]
x =ncm12_desc[ncm12_desc["Posición"].isin(sesgos_neg.index)]
# x.to_csv("partidas_con_sesgo_precio_negativo.csv", sep=";")

len(sesgos[sesgos<0])/len(sesgos)
len(sesgos[sesgos>0])

## precios maximo, minimo y mediano

precios = join_impo_clae_bec_bk.groupby(["HS6_d12"], as_index = False).agg({"precio": ["median", "mean", lambda x: x.std()/x.mean(),
                                                                                       "max", "min", lambda x: x.max()- x.min()]}) 
precios.columns.levels[1]

precios.columns.set_levels([ "precio_simple_med" ,"precio_simple_avg", "precio_simple_cv", "precio_simple_max", "precio_simple_min", "precio_simple_rang","HS6_d12" ] , level= 1, inplace  = True)
precios.columns = precios.columns.droplevel(level=0)

indicadores_all = pd.merge(indicadores, precios, how = "inner", left_on ="HS6_d12", right_on = "HS6_d12")

#medidas de homogeneidad al interior de los bienes (sd, cv, etc)
# diferencia entre precio unitario mas alto y el mas bajo (rango) y cantidad por empresa

#cluster 
import sklearn


## productos que solo los trae una empresa
indicadores.sort_values("cuit")
productos_una_empresa = indicadores[indicadores["cuit"]==1]
x = join_impo_clae_bec_bk[join_impo_clae_bec_bk["HS6_d12"].isin(productos_una_empresa.index) ]

indicadores.sort_values("cant_x_empresa",ascending = False)

#electric generating set
impo_x = join_impo_clae_bec_bk[join_impo_clae_bec_bk["HS6"] == 850213]
impo_x["precio"].describe()
impo_x["precio"].hist(bins = 100) 
impo_x["cantidad"].hist(bins = 100) 

#Pumps; for liquids, fitted or designed to be fitted with a measuring device, other than pumps for dispensing fuel or lubricants
impo_x = join_impo_clae_bec_bk[join_impo_clae_bec_bk["HS6"] == 841319]
impo_x["precio"].describe()
impo_x["precio"].hist(bins = 100) 
impo_x["cantidad"].hist(bins = 100) 

#camiones
impo_camiones = impo_d12[impo_d12["HS6"] == 870421]
impo_camiones= impo_camiones.groupby(["cuit", "NOMBRE"])[["cantidad", "valor"]].sum().sort_values("cantidad", ascending= False)
impo_camiones["cantidad"].describe()
impo_camiones["cantidad"].hist(bins = 100) 

impo_ford = join_impo_clae_bec_bk[join_impo_clae_bec_bk["cuit"] == "30678519681"].sort_values("valor", ascending = False)

#cosechadoras
impo_x = impo_d12[impo_d12["HS6"] == 843351]
impo_x= impo_x.groupby(["cuit", "NOMBRE"])[["cantidad", "valor"]].sum().sort_values("cantidad", ascending= False)
impo_x["cantidad"].describe()
impo_x["cantidad"].hist(bins = 100) 
sns.displot(impo_x["cantidad"])

#tractores
impo_x = impo_d12[impo_d12["HS6"] == 870120]
impo_x= impo_x.groupby(["cuit", "NOMBRE"])[["cantidad", "valor"]].sum().sort_values("cantidad", ascending= False)
impo_x["cantidad"].describe()
impo_x["cantidad"].hist(bins = 100) 

#camion con pala 
impo_x = impo_d12[impo_d12["HS6"] == 842951]
impo_x= impo_x.groupby(["cuit", "NOMBRE"])[["cantidad", "valor"]].sum().sort_values("cantidad", ascending= False)
impo_x["cantidad"].describe()
impo_x["cantidad"].hist(bins = 100) 


#############################################
#         ASIGNACIÓN y MATRIZ               #
#############################################

#matriz_sisd = def_insumo_matriz(join_final)
#matriz_sisd.to_csv("../data/resultados/matriz_pesada_12d_6act.csv", index= False)
matriz_sisd = pd.read_csv("../data/resultados/matriz_pesada_12d_6act.csv")


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



