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

import re

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
impo_d12 = pd.read_csv("../data/IMPO_17_feature.csv")
# impo_d12 = pd.read_csv("../data/IMPO_2017_12d.csv")
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
# EDA BEC5
# =============================================================================
def destinacion_limpio(x):
    if re.search("PARA TRANSF|C/TRANS|P/TRANS|RAF|C/TRNSF|ING.ZF INSUMOS", x)!=None:
        return "C/TR"
    elif re.search("S/TRAN|SIN TRANSF|INGR.ZF BIENES", x)!=None:
        return "S/TR"
    elif re.search("CONS|ING.ZF MERC", x)!=None:
        # return "CONS"
        return "CONS&Otros"
    else: 
        # return "Otro"
        return "CONS&Otros"


impo_d12["dest_clean"] = impo_d12["destinacion"].apply(lambda x: destinacion_limpio(x))
impo_d12["dest_clean"].value_counts()

impo_bec = pd.merge(impo_d12, bec[["HS6", "BEC5EndUse" ]], how= "left" , left_on = "HS6", right_on= "HS6" )



bec["BEC5EndUse"].value_counts().sum()
bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]["BEC5EndUse"].value_counts()#.sum()
bec[bec["BEC5EndUse"].str.startswith("INT", na = False)]["BEC5EndUse"].value_counts()#.sum()
bec[bec["BEC5EndUse"].str.startswith("CONS", na = False)]["BEC5EndUse"].value_counts()#.sum()


# =============================================================================
# VENN BK
# =============================================================================
impo_bec_bk = impo_bec[impo_bec["BEC5EndUse"].str.startswith("CAP", na = False)] 
impo_bec_bk ["dest_clean"].value_counts()#.sum()

filtro1st =  impo_bec_bk [impo_bec_bk ["dest_clean"] == "S/TR"] 
filtro1ct =  impo_bec_bk [impo_bec_bk ["dest_clean"] == "C/TR"] 
filtro1co =  impo_bec_bk [impo_bec_bk ["dest_clean"] == "CONS&Otros"] 

len(impo_bec_bk )  == ( len(filtro1st) +len(filtro1ct) + len(filtro1co) ) 

# Filtros de conjuntos
set_st = set(filtro1st["HS6_d12"])
set_ct = set(filtro1ct["HS6_d12"])
set_co = set(filtro1co["HS6_d12"])

filtro_a = set_st - set_co - set_ct 
filtro_b = set_ct - set_st - set_co 
filtro_c = set_co - set_ct -set_st 

filtro_d = (set_st & set_co) - (set_ct )
filtro_e = (set_ct & set_co) - ( set_st)
filtro_f = (set_ct & set_st) - set_co 
filtro_g = set_ct & set_st & set_co 

dest_a = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_a  )]
dest_b = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_b  )]
dest_c = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_c  )]
dest_d = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_d  )]
dest_e = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_e  )]
dest_f = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_f  )]
dest_g = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_g )]

(len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) +len(dest_b)+ len(dest_c)) ==len(impo_bec_bk)


# =============================================================================
# VENN CI
# =============================================================================
impo_bec_ci = impo_bec[impo_bec["BEC5EndUse"].str.startswith("INT", na = False)] 
impo_bec_ci ["dest_clean"].value_counts()#.sum()

filtro1st =  impo_bec_ci [impo_bec_ci ["dest_clean"] == "S/TR"] 
filtro1ct =  impo_bec_ci [impo_bec_ci ["dest_clean"] == "C/TR"] 
filtro1co =  impo_bec_ci [impo_bec_ci ["dest_clean"] == "CONS&Otros"] 

len(impo_bec_ci )  == ( len(filtro1st) +len(filtro1ct) + len(filtro1co) ) 

# Filtros de conjuntos
set_st = set(filtro1st["HS6_d12"])
set_ct = set(filtro1ct["HS6_d12"])
set_co = set(filtro1co["HS6_d12"])

filtro_a = set_st - set_co - set_ct 
filtro_b = set_ct - set_st - set_co 
filtro_c = set_co - set_ct -set_st 

filtro_d = (set_st & set_co) - (set_ct )
filtro_e = (set_ct & set_co) - ( set_st)
filtro_f = (set_ct & set_st) - set_co 
filtro_g = set_ct & set_st & set_co 

dest_a = impo_bec_ci[impo_bec_ci["HS6_d12"].isin( filtro_a  )]
dest_b = impo_bec_ci[impo_bec_ci["HS6_d12"].isin( filtro_b  )]
dest_c = impo_bec_ci[impo_bec_ci["HS6_d12"].isin( filtro_c  )]
dest_d = impo_bec_ci[impo_bec_ci["HS6_d12"].isin( filtro_d  )]
dest_e = impo_bec_ci[impo_bec_ci["HS6_d12"].isin( filtro_e  )]
dest_f = impo_bec_ci[impo_bec_ci["HS6_d12"].isin( filtro_f  )]
dest_g = impo_bec_ci[impo_bec_ci["HS6_d12"].isin( filtro_g )]

(len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) +len(dest_b)+ len(dest_c)) ==len(impo_bec_ci)


# =============================================================================
# VENN CONS
# =============================================================================
impo_bec_cons = impo_bec[impo_bec["BEC5EndUse"].str.startswith("CONS", na = False)] 
impo_bec_cons ["dest_clean"].value_counts()#.sum()

filtro1st =   impo_bec_cons [ impo_bec_cons ["dest_clean"] == "S/TR"] 
filtro1ct =   impo_bec_cons [ impo_bec_cons ["dest_clean"] == "C/TR"] 
filtro1co =   impo_bec_cons [ impo_bec_cons ["dest_clean"] == "CONS&Otros"] 

len( impo_bec_cons )  == ( len(filtro1st) +len(filtro1ct) + len(filtro1co) ) 

# Filtros de conjuntos
set_st = set(filtro1st["HS6_d12"])
set_ct = set(filtro1ct["HS6_d12"])
set_co = set(filtro1co["HS6_d12"])

filtro_a = set_st - set_co - set_ct 
filtro_b = set_ct - set_st - set_co 
filtro_c = set_co - set_ct -set_st 

filtro_d = (set_st & set_co) - (set_ct )
filtro_e = (set_ct & set_co) - ( set_st)
filtro_f = (set_ct & set_st) - set_co 
filtro_g = set_ct & set_st & set_co 

dest_a =  impo_bec_cons[ impo_bec_cons["HS6_d12"].isin( filtro_a  )]
dest_b =  impo_bec_cons[ impo_bec_cons["HS6_d12"].isin( filtro_b  )]
dest_c =  impo_bec_cons[ impo_bec_cons["HS6_d12"].isin( filtro_c  )]
dest_d =  impo_bec_cons[ impo_bec_cons["HS6_d12"].isin( filtro_d  )]
dest_e =  impo_bec_cons[ impo_bec_cons["HS6_d12"].isin( filtro_e  )]
dest_f =  impo_bec_cons[ impo_bec_cons["HS6_d12"].isin( filtro_f  )]
dest_g =  impo_bec_cons[ impo_bec_cons["HS6_d12"].isin( filtro_g )]

(len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) +len(dest_b)+ len(dest_c)) ==len( impo_bec_cons)


len(impo_bec_ci) + len(impo_bec_bk) + len(impo_bec_cons)


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

#productos q los trae una sola empresa
n_emp_1 = indicadores[(indicadores["n_emp_importadoras"]==1 ) & (indicadores["precio_emp_avg"] >indicadores["precio_emp_avg"].median()  ) ]
corr = n_emp_1.drop(["unidad_medida", "HS6_d12", "n_emp_importadoras", "control_precio"], axis = 1).corr()

# correlaciones
corr  = indicadores.loc[ indicadores["n_emp_importadoras"]==1, [ 'cant_importada', 'n_emp_importadoras',
                     'valor_total','cant_x_empresa', 'precio_emp_avg',]].drop("n_emp_importadoras", axis = 1).corr()

corr  = indicadores.loc[:, [ 'cant_importada', 'n_emp_importadoras',
                     'valor_total','cant_x_empresa', 'precio_emp_avg']].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0, annot=True,#  mask=mask,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# tabla con n_empresa == 1 , agregarle las 6 actividades
#kmeans entre cantidad_x_emp y precio unitario
#cluster 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D


scaler = StandardScaler()
data = indicadores[[ 'n_emp_importadoras', "cant_x_empresa", 'precio_emp_avg']]
data_scale = pd.DataFrame (scaler.fit_transform(data), columns = data.columns )

sns.pairplot(data_scale ,kind='scatter')
data_scale.hist()

# kmeans = KMeans( n_clusters=2).fit(data_scale)
kmeans = MiniBatchKMeans( n_clusters=2).fit(data_scale)
centroids = kmeans.cluster_centers_


# Predicting the clusters
labels = kmeans.predict(data_scale)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['red','green']
asignar=[]
for row in labels:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_scale.iloc[:, 0], data_scale.iloc[:, 1], data_scale.iloc[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
ax.set_xlabel('Cantidad de empresas importadoras', rotation=150)
ax.set_ylabel('Productos por empresa')
ax.set_zlabel('Precio promedio ponderado',  rotation=60)

f2 = data_scale['precio_emp_avg'].values
# f2 = data_scale['cant_x_empresa'].values
f1 = data_scale['n_emp_importadoras'].values
 
plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 2], C[:, 1], marker='*', c=["red", "green"], s=1000)
plt.show()

pd.value_counts(labels)

data_scale["cluster"] =labels


# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['red','green','blue','cyan','yellow']
asignar=[]
for row in labels:
    asignar.append(colores[row])
 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

#PRECIOS
# Precio unitario (todos los productos)
join_impo_clae_bec_bk["precio"] =join_impo_clae_bec_bk["valor"]/join_impo_clae_bec_bk["cantidad"]  

precios["precio__skew"].hist(bins = 100) 
plt.title("Histogramas de las asimetrias de los precios unitarios")

## precios maximo, minimo y mediano
precios = join_impo_clae_bec_bk.groupby(["HS6_d12"], as_index = False).agg({"precio": ["median", "mean", lambda x: x.std()/x.mean(),
                                                                                       "max", "min", lambda x: x.max()- x.min(),
                                                                                       lambda x: x.skew(axis=0, skipna=False) ]}) 


precios.columns.levels[1]

precios.columns.set_levels([ "precio_simple_med" ,"precio_simple_avg", "precio_simple_cv", "precio_simple_max", "precio_simple_min", "precio_simple_rang", "precio_simple_skew", "HS6_d12" ] , level= 1, inplace  = True)
precios.columns = precios.columns.droplevel(level=0)



precios_empresa = indicadores_raw.groupby(["HS6_d12"], as_index = False).agg({"precio_empresa": ["median", "mean", lambda x: x.std()/x.mean(),
                                                                                       "max", "min", 
                                                                                       lambda x: x.quantile(q= .25),
                                                                                       lambda x: x.quantile(q= .75),
                                                                                       lambda x: (x.max()- x.min())/x.mean(),
                                                                                       lambda x: x.skew(axis=0, skipna=False) ]}) 

precios_empresa.columns.levels[1]
precios_empresa.columns.set_levels([ "precio_emp_med" ,"precio_emp_avg", "precio_emp_cv", "precio_emp_max", "precio_emp_min", "precio_emp_25", "precio_emp_75", "precio_emp_rang", "precio_emp_skew", "HS6_d12" ] , level= 1, inplace  = True)
precios_empresa.columns = precios_empresa.columns.droplevel(level=0)

corr = precios_empresa.drop("HS6_d12", axis =1).corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0, annot=True, mask=mask,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

x = pd.merge(join_impo_clae_bec_bk [['NOMBRE', 'HS6_d12', 'valor', 'unidad_medida', 
                                             'cantidad', 'HS6', 'Descripción Completa', "precio"]], precios_empresa, how = "left", left_on="HS6_d12",  right_on="HS6_d12")
separacion = pd.merge(x, indicadores[['HS6_d12', 'n_emp_importadoras',  'cant_x_empresa']], 
                      how = "left", left_on = "HS6_d12", right_on = "HS6_d12")


sep_partes = separacion[separacion["precio"] < separacion["precio_emp_25"] ]
sep_bk = separacion[separacion["precio"] > separacion["precio_emp_25"] ]


indicadores_all = pd.merge(indicadores, precios, how = "inner", left_on ="HS6_d12", right_on = "HS6_d12")

#medidas de homogeneidad al interior de los bienes (sd, cv, etc)
# diferencia entre precio unitario mas alto y el mas bajo (rango) y cantidad por empresa



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



