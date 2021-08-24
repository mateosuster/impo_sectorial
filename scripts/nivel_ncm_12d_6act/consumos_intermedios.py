# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:38:55 2021

@author: mateo
"""

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
# impo_d12 = pd.read_csv("../data/IMPO_2017_12d.csv")
impo_d12  = pd.read_csv("../data/IMPO_17_feature.csv")
# impo_d12  = pd.read_excel("../data/IMPO_17_feature.xlsx")
# impo_d12.to_csv("../data/IMPO_17_feature.csv")

#impo_17 = pd.read_csv(  "../data/IMPO_2017.csv", sep=";")
clae = pd.read_csv( "../data/clae_nombre.csv")
comercio = pd.read_csv("../data/comercio_clae.csv", encoding="latin1")
#cuit_clae = pd.read_csv( "../data/cuit 2017 impo_con_actividad.csv")
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")
bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018.csv")
bec_to_clae = pd.read_csv("../data/bec_to_clae.csv")

#diccionario ncm12d
ncm12_desc = pd.read_csv("../data/NCM 12d.csv", sep=";")#.rename(columns = {"Descripción Completa": "descripcion"})
ncm12_desc_split = pd.concat([ncm12_desc.iloc[:,0], pd.DataFrame(ncm12_desc['Descripción Completa'].str.split('//', expand=True))], axis=1)

# parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")

# bce_cambiario = pd.read_csv("../data/balance_cambiario.csv", skiprows = 3, error_bad_lines=False, sep= ";", na_values =['-'])
# isic = pd.read_csv("../data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )
# dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")
#STP
dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")
# dic_stp.to_csv("../data/bsk-prod-clasificacion.csv", index=False)

#############################################
#           preparación bases               #
#############################################

# predo_impo_17(impo_17)
impo_d12  = predo_impo_12d(impo_d12, ncm12_desc)# , kilos = True)
letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae)
bec_bk = predo_bec_bk(bec, bec_to_clae)
dic_stp = predo_stp(dic_stp )


#############################################
#                joins                      #
#############################################
join_impo_clae = def_join_impo_clae(impo_d12, cuit_empresas)
join_impo_clae_bec_bk = def_join_impo_clae_bec(join_impo_clae, bec_bk)
# join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio)


#############################################
#           Tabla de contingencia           #
#              producto-sector              #
############################################
# tabla_contingencia = def_contingencia(join_impo_clae_bec_bk_comercio)

#############################################
#      ponderación por ncm y letra          #
#############################################

# join_impo_clae_bec_bk_comercio_pond = def_join_impo_clae_bec_bk_comercio_pond(join_impo_clae_bec_bk_comercio, tabla_contingencia)

#join_final = def_calc_pond(join_impo_clae_bec_bk_comercio_pond,tabla_contingencia)
#join_final.to_csv("../data/resultados/impo_con_ponderaciones_12d_6act.csv", index=False)
#join_final = pd.read_csv("../data/resultados/impo_con_ponderaciones_12d_6act.csv")

# filtro = ["HS6", "CUIT_IMPOR", "valor", "letra1", "letra2", "letra3", 
# "vta_bk", "vta_sec", "vta_bk2", "vta_sec2", "vta_bk3", "vta_sec3", 
# "letra1_pond", "letra2_pond", "letra3_pond"]
# join_final = join_final.sort_values("HS6")[filtro]

# =============================================================================
# Destino sectorial de las partidas
# =============================================================================
ncm_sector = pd.read_csv("../nivel_clae_letra/resultados/asignacion_ncm.csv")
ncm_sector["hs6"] = ncm_sector["hs6"].astype(int)

# =============================================================================
# 1er filtro: STP para agro y transporte 
# =============================================================================

#arreglar transporte (camiones ) y agro (cosechadoras). revisar donde caen los de desc gral: "Maq agr y forestal"

#filtro transporte
stp_trans= dic_stp[dic_stp["utilizacion"] == "Transporte"]

# filtros agro (son iguales)
stp_agro = dic_stp[dic_stp["demanda"].str.contains("agríc", case =False)]
stp_agro = dic_stp[dic_stp["desc_gral"].str.contains("Maquinaria agropecuaria y forestal")]

# porcentaje relativo de las asignaciones
ncm_sector_perc = ncm_sector.set_index("hs6")
ncm_sector_perc = ncm_sector_perc.div(ncm_sector_perc.sum(axis=1), axis = 0).mul(100).round(3).reset_index()

# destino de los bienes de la stp 
stp_asig = pd.merge(dic_stp[["NCM", "desc", "demanda"]], ncm_sector_perc, how= "left", left_on = "NCM", right_on = "hs6" )
stp_agro_asig = pd.merge(stp_agro[["NCM", "desc", "demanda"]], ncm_sector_perc, how= "left", left_on = "NCM", right_on = "hs6" ).drop("hs6", axis=1)
stp_trans_asig = pd.merge(stp_trans[["NCM", "desc", "demanda"]], ncm_sector_perc, how= "left", left_on = "NCM", right_on = "hs6" ).drop("hs6", axis=1)

# filtrado 1
filtro1 = join_impo_clae_bec_bk[~join_impo_clae_bec_bk["HS6"].isin( pd.concat([stp_agro["NCM"] , stp_trans["NCM"]]))]

(len(filtro1)/len(join_impo_clae_bec_bk)-1)*100
len(filtro1)
len(join_impo_clae_bec_bk)

# =============================================================================
# 2do filtro: STP entre específicos y generales
# =============================================================================
#frecuencia 
dic_stp["utilizacion"].value_counts()

# vectores filtros
stp_general = dic_stp[dic_stp["utilizacion"]=="General"] # =~ CI
stp_especifico = dic_stp[dic_stp["utilizacion"]=="Específico"] # =~ BK

# filtrado 2
filtro2 = filtro1[~filtro1["HS6"].isin(stp_especifico["NCM"] )]
# filtro2b = filtro1[filtro1["HS6"].isin(stp_general["NCM"] )]
(len(filtro2)/len(filtro1)-1)*100
len(filtro2)

#magnitud del problema
filtro2.valor.sum()/impo_d12.valor.sum()
filtro2.valor.sum()/join_impo_clae_bec_bk.valor.sum()

# partes y piezas
# partes_stp = stp_asig[stp_asig["desc"].str.contains("parte", case = False) ]
# accesorios_stp = stp_asig[stp_asig["desc"].str.contains("accesorio", case = False) ]

# =============================================================================
# 3er filtro: Variable Destinacion
# =============================================================================
pd.unique(impo_d12["destinacion"])

partidas_dest = join_impo_clae_bec_bk[join_impo_clae_bec_bk["destinacion"].str.contains("temp|RAF|C/TR", case=False)]
partidas_dest = partidas_dest [~partidas_dest ["destinacion"].str.contains("S/TRA|SIN TRA", case=False)]

partidas_dest .destinacion.value_counts(normalize= True)

filtro3 = filtro2[filtro2["HS6"].isin(partidas_dest["HS6"]) ]
(len(filtro3)/len(filtro2)-1)*100
len(filtro3)

x = filtro3 .groupby(["HS6_d12", "descripcion",  "destinacion"], as_index= False).agg("size")


###################### pruebas
# impo_d12.destinacion.value_counts(normalize= True)
# join_impo_clae_bec_bk.destinacion.value_counts(normalize= True)

# impo_d12[impo_d12["destinacion"].str.contains("bienes de capital", case=False)]["valor"].sum()/filtro_stp ["valor"].sum()
# impo_d12[impo_d12["destinacion"].str.contains("temporaria", case=False)]["valor"].sum()/impo_d12["valor"].sum()
# filtro_stp[filtro_stp["destinacion"].str.contains("temporaria", case=False)]["valor"].sum()/filtro_stp ["valor"].sum()

# # prueba_dest = filtro_stp.groupby(["HS6_d12", "descripcion",  "destinacion"])["valor"].agg(  "sum")
# prueba_dest = join_impo_clae_bec_bk.groupby(["HS6_d12", "descripcion",  "destinacion"])["valor"].agg(  "sum")
# prueba_relativ =prueba_dest.groupby(level=0).apply(lambda x : x / float(x.sum()))


# x = join_impo_clae_bec_bk.groupby(["HS6_d12", "descripcion",  "destinacion"], as_index= False).agg("size")
# x[x["size"]] = 1

# y = x[x["destinacion"].str.contains("C/TR", case=False)]

# x = join_impo_clae_bec_bk.groupby(["HS6_d12", "descripcion"], as_index= False)["destinacion"].agg("count")
# y = x[x["destinacion"].str.contains("temporaria|RAF", case=False)]

# =============================================1================================
# 4to filtro: Metricas 
# =============================================================================
# probar con cantidad importada (en sus respectivas unidades declaradas y en kilos, y en peso unitario)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D

######### 
# loop 
cols = ["HS6_d12", "descripcion", "destinacion", "uni_decl", "cant_decl"]
data = filtro3[cols]
data["groupID"]= data.groupby(["HS6_d12", "destinacion", "uni_decl"]).ngroup()
data["ID"]= range(len(data))

data[["HS6_d12","descripcion",  "destinacion", "uni_decl"]].value_counts(ascending=False)

filtro_raw = data.groupby(["HS6_d12","descripcion",  "destinacion", "uni_decl","groupID"],as_index=False).size()
filtro_mayor1 = filtro_raw[filtro_raw["size"]>1]
filtro_menor1= filtro_raw[filtro_raw["size"]==1]
filtro = pd.unique(filtro_mayor1["groupID"])
len(filtro )

from tqdm import tqdm

clusters = pd.DataFrame(columns= ["groupID", "cant_decl_clu", "cluster", "ID"])
for i in tqdm(filtro):
    # print(i)
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=1)
    # x = data[data["groupID"]== filtro[0] ]["cant_decl"].to_numpy() 
    x = np.array( data[data["groupID"]== i ]["cant_decl"] ).reshape(-1,1)
    cluster = kmeans.fit(x)
    label = cluster.labels_
    to_append = pd.DataFrame({ "groupID":i,  "cant_decl_clu": x.reshape(len(label)), "cluster": label})
    clusters = pd.concat([clusters, to_append], axis = 0)

clusters.sort_values(by = ["groupID", "cant_decl_clu"], inplace = True)
clusters["ID"]= range(len(clusters))    

data_clusterizada = data[data["groupID"].isin(filtro)].sort_values(by = ["groupID", "cant_decl"] )
data_clusterizada["ID"]= range(len(data_clusterizada))
data_cluster = pd.merge(data_clusterizada, clusters, left_on = ["groupID", "ID"],  right_on = ["groupID", "ID"], how = "inner")
np.mean(data_cluster["cant_decl"]/data_cluster["cant_decl_clu"])

# LISTO CLUSTER !

#######################################################################################

#############
x = filtro3.groupby(["HS6_d12", "destinacion", "uni_decl"], as_index= False)["cant_decl"]


def apply_kmeans_on_each_category(df):
    not_na_mask = df.notna()
    
    embedding = df.loc[not_na_mask]
    n_clusters = 2
    # n_clusters = int(not_na_mask.sum()/2)

    op = pd.Series([np.nan] * len(df), index=df.index)
    if n_clusters > 1:
        df['cluster'] = np.nan
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(embedding.tolist())
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding.tolist())
        op.loc[not_na_mask] = kmeans.labels_.tolist()
    return op
# df_test['clusters'] = df_test.groupby('A')['D'].transform(apply_kmeans_on_each_category)
filtro3["cluster"] = filtro3.groupby(["HS6_d12", "destinacion", "uni_decl"], as_index= False)["cant_decl"].transform(apply_kmeans_on_each_category)
data["cluster"] = data.groupby(["HS6_d12", "destinacion", "uni_decl"], as_index= False)["cant_decl"].apply(apply_kmeans_on_each_category)

scaler = StandardScaler()
cant_scale = pd.DataFrame(scaler.fit_transform(filtro3[["cant_decl"]])).rename(columns = {0: "cant_decl"})
cant_scale["id"] = range(len(cant_scale)) 

data_sel = filtro3[["HS6_d12", "destinacion", "uni_decl"]]
data_sel["id"] = range(len(data_sel))

data = pd.merge(data_sel, cant_scale, left_on = "id", right_on ="id", how ="inner" ).drop("id", axis =1)

X = np.array(data).reshape(-1,1)  
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

kmeans = MiniBatchKMeans( n_clusters=2).fit(data.groupby(["HS6_d12", "destinacion", "uni_decl"])) 
kmeans = KMeans( n_clusters=2).fit(data.groupby(["HS6_d12", "destinacion", "uni_decl"])) 

#####

def cluster(X):
    k_means = KMeans(n_clusters=2).fit(X)
    # k_means = MiniBatchKMeans(n_clusters=2).fit(X)
    return X.groupby(k_means.labels_)\
            .transform('mean').sum(1)\
            .rank(method='dense').sub(1)\
            .astype(int).to_frame()
    # return k_means

cluster(np.array(data["cant_decl"]).reshape(1,-1) )

data['Cluster_id'] = data.groupby(["HS6_d12", "destinacion", "uni_decl"])["cant_decl"].apply(cluster)
data['Cluster_id'] = data.groupby(["HS6_d12", "destinacion", "uni_decl"])["cant_decl"].transform(cluster)


data.sort_values("groupID")

data['Cluster_id'] = data.groupby(["HS6_d12", "destinacion", "uni_decl"]).apply(lambda x : cluster(X))


df['Cluster_cat'] = df['Cluster_id'].map(mapping)

####
filtro3["cluster"] = filtro3.groupby(["HS6_d12", "destinacion", "uni_decl"], as_index= False)["cant_decl"].transform()



np.array([x, label])

type(label)
type(x)
len(label)
type(x)
label.shape
x.reshape(len(label))

x = kmeans.fit(data[data["groupID"]== filtro[0]]["cant_decl"])



# combinacion de unidades 
x = impo_d12.groupby(["uni_est", "uni_decl"]).size().reset_index().rename(columns={0:'count'}).sort_values("count", ascending = False)
x["frec_relativa"] = x["count"]/(x["count"].sum() )

x = impo_d12[impo_d12["HS6_d12"].isin(n_emp_1["HS6_d12"])].groupby(["unidad_est", "unidad_decl"]).size().reset_index().rename(columns={0:'count'}).sort_values("count", ascending = False)
x["frec_relativa"] = x["count"]/(x["count"].sum() )


y = impo_d12[(impo_d12["HS6_d12"].isin(n_emp_1["HS6_d12"])) & (impo_d12["unidad_est"]=="Unidad") & (impo_d12["unidad_decl"]=="Unidad") ]
# y = impo_d12[(impo_d12["HS6_d12"].isin(n_emp_1["HS6_d12"])) ]
y.valor.sum() / impo_d12.valor.sum()


# =============================================================================
# Resultado filtro
# =============================================================================
observaciones_filtros = pd.DataFrame([len(join_impo_clae), len(join_impo_clae_bec_bk), len(filtro1), len(filtro2), len(filtro3)], columns= ["n"])
observaciones_filtros["rel_al_tot"] =  observaciones_filtros["n"]/len(join_impo_clae_bec_bk)


# =============================================================================
# construccion de  features
# =============================================================================
join_impo_clae_bec_bk[join_impo_clae_bec_bk["HS6_d12"]== "84241000910C" ]
x = indicadores[indicadores["HS6_d12"]== "84241000910C" ]
n_emp_1 [n_emp_1 ["HS6_d12"]== "84241000910C" ]

#cantidad de empresas que importan la partida y cantidad y valor importada total x partida
indicadores_raw= join_impo_clae_bec_bk.groupby(["cuit", "uni_est", "uni_decl",  "HS6_d12"], as_index = False).agg({ "cant_est":"sum", "valor": "sum"
                                                                                                         , "kilos": "sum"})

indicadores_raw["precio_empresa" ] = indicadores_raw.apply( lambda x: x.valor / x.cant_est, axis = 1 )

precio_emp_avg = indicadores_raw.groupby( ["uni_est", "HS6_d12"], as_index= False)["precio_empresa"].mean().rename(columns = {"precio_empresa": "precio_emp_avg"})

indicadores = indicadores_raw.reset_index().groupby([ "uni_est", "HS6_d12"], as_index = False).agg({"cuit": "count", "cant_est":"sum", "valor": "sum"
                                                                                                       , "kilos": "mean"
                                                                                                       })
indicadores .rename(columns= {"cuit": "n_emp_importadoras", 
                            "cant_est": "cant_importada",
                            "valor": "valor_total"}, inplace = True)

indicadores["kilo_x_uni"] =indicadores["kilos"]/indicadores["cant_importada"]

# Productos por empresa
indicadores["cant_x_empresa"] = indicadores["cant_importada"]/indicadores["n_emp_importadoras"]

#precio
indicadores["precio_sin_ponderar"] = indicadores["valor_total"]/indicadores["cant_importada"]

indicadores = pd.merge(indicadores, precio_emp_avg.drop("uni_est", axis =1) , how = "left", left_on = "HS6_d12",right_on = "HS6_d12")

indicadores["control_precio"] = indicadores["precio_sin_ponderar"]/indicadores["precio_emp_avg"] -1 
indicadores["control_precio"].hist()
indicadores["kilos"].hist(bins = 100)

indicadores = pd.merge(indicadores, impo_d12[[ "HS6_d12", "descripcion"]].drop_duplicates(), how = "left", left_on = "HS6_d12", right_on = "HS6_d12")


indicadores_un = indicadores[indicadores["unidad_est"] == "Unidad"]
indicadores_kg = indicadores[indicadores["unidad_est"] == "Kilogramo"]

np.unique(indicadores[["unidad_est", unidad_]])


sns.scatterplot(x = "kilos", y =  "precio_emp_avg", data = indicadores ) 
sns.scatterplot(x = "kilo_x_uni", y =  "precio_emp_avg", data = indicadores_un ) 


#productos q los trae una sola empresa
n_emp_1 = indicadores[(indicadores["n_emp_importadoras"]==1 )  ]


n_emp_1 = indicadores_un[(indicadores_un["n_emp_importadoras"]==1 )  ]
n_emp_1 = n_emp_1[ (n_emp_1["precio_emp_avg"] >n_emp_1["precio_emp_avg"].median()  )  ]

x = n_emp_1[(n_emp_1["kilo_x_uni"]< 25000) & (n_emp_1["precio_emp_avg"]< 10**6) ]
x = n_emp_1[(n_emp_1["kilo_x_uni"]< 5000)] 
x = x[(x["precio_emp_avg"]< (0.4*10)**6) ]
sns.scatterplot(x = "kilo_x_uni", y =  "precio_emp_avg", data = x ) 

corr = n_emp_1.drop(["unidad_est", "HS6_d12", "n_emp_importadoras", "control_precio"], axis = 1).corr()
corr = x.drop(["unidad_est", "HS6_d12", "n_emp_importadoras", "control_precio"], axis = 1).corr()

sns.scatterplot(x = "cant_x_empresa", y =  "precio_emp_avg", data = n_emp_1) 
sns.scatterplot(x = "kilos", y =  "valor_total", data = n_emp_1 ) 
sns.scatterplot(x = "kilos", y =  "precio_emp_avg", data = n_emp_1 ) 
sns.scatterplot(x = "kilo_x_uni", y =  "precio_emp_avg", data = n_emp_1 ) 

n_emp_1[n_emp_1["HS6_d12"]== "8424"]
n_emp_1["valor_total"].sum()
n_emp_1["precio_emp_avg"].hist(bins = 100)

n_emp_1["precio_emp_avg"].median()
menor_cuantil = n_emp_1[n_emp_1["precio_emp_avg"] <= n_emp_1["precio_emp_avg"].quantile(q = .25)]
len(menor_cuantil )


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
kmeans.labels_

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
                                             'cantidad', 'HS6', 'descripcion', "precio"]], precios_empresa, how = "left", left_on="HS6_d12",  right_on="HS6_d12")
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
