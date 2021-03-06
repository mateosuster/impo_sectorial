
# =============================================================================
# Directorio de trabajo y librerias
# =============================================================================
import os 
# os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
# os.chdir("C:/Users/igalk/OneDrive/Documentos/laburo/CEP/procesamiento impo/nuevo1/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

from def_bases_nivel_ncm_12d_6act import *
from def_procesamiento_nivel_ncm_12d_6act import *
from def_matriz_nivel_ncm_12d_6act import *
from def_pre_visualizacion_nivel_ncm_12d_6act import *

#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
start = datetime.datetime.now()
datos = pd.read_csv("../data/heavys/datos_clasificados_modelo_all_data_21oct.csv", sep = ";")

#auxiliares
clae = pd.read_csv( "../data/clae_nombre.csv")
hs_to_isic = pd.read_csv("../data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")
dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")
vector_comercio_bk = pd.read_csv("../data/vector_de_comercio_clae_bk.csv", sep = ";").drop("Unnamed: 0", 1)
vector_comercio_ci = pd.read_csv("../data/vector_de_comercio_clae_ci.csv", sep = ";")#.drop(["letra", "clae6_desc"] , axis = 1)
clae_to_ciiu = pd.read_excel("../data/Pasar de CLAE6 a CIIU3.xlsx")

dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")
dic_propio = predo_dic_propio(clae_to_ciiu, dic_ciiu,clae)

#mectra
mectra_pond = pd.read_csv("../data/resultados/agr_csv_files/mectra_agr_2021-12-30.csv")
mectra_pond = mectra_pond[mectra_pond["anio"] ==2017].drop("anio", 1).rename(columns = {"propio_letra_2":"sd"})

#############################################
#           preparación bases               #
#############################################
letras = predo_sectores_nombres(clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae) #meter loop aca
dic_stp = predo_stp(dic_stp)

datos = preprocesamiento_datos(datos, dic_propio)
datos_bk , datos_bk_sin_picks, bk_picks = asignacion_stp_BK(datos, dic_stp)
datos_ci = filtro_ci(datos)


#############################################
#         BK                                #
#############################################
# non pick ups
datos_bk_comercio = def_join_comercio(datos_bk_sin_picks , vector_comercio_bk)  
tabla_contingencia = def_contingencia(datos_bk_comercio , datos) #Tabla de contingencia producto-sector   
datos_bk_comercio_pond= def_ponderaciones(datos_bk_comercio, tabla_contingencia, ci = False) #ponderación por ncm y letra
matriz_sisd_insumo = def_asignacion_sec(datos_bk_comercio_pond) ##asignación por probabilidad de G-bk (insumo para la matriz)
asign_pre_matriz_bk= def_asignacion_prob(matriz_sisd_insumo)

# picks ups
asign_pre_matriz_pick  = def_asignacion_picks(bk_picks, mectra_pond)
# asign_pre_matriz_pick .groupby("sd")["valor_pond"].sum().sort_values()

# ALL
asign_pre_matriz_bk = pd.concat([asign_pre_matriz_bk, asign_pre_matriz_pick]) #igal 17/2
matriz_sisd_bk = to_matriz(asign_pre_matriz_bk, 0) #matriz SISD
matriz_hssd_bk  = pd.pivot_table(asign_pre_matriz_bk, values='valor_pond', index=['hs6_d12'], columns=['sd'], aggfunc=np.sum, fill_value=0)

#############################################
#             CI                          #
#############################################
datos_ci_comercio  = def_join_comercio(datos_ci , vector_comercio_ci, ci = True)
tabla_contingencia_ci = def_contingencia(datos_ci_comercio  , datos)
datos_ci_comercio_pond = def_ponderaciones(datos_ci_comercio , tabla_contingencia_ci, ci = True)
matriz_sisd_insumo_ci = def_asignacion_sec(datos_ci_comercio_pond, ci = True)
asign_pre_matriz_ci= def_asignacion_prob(matriz_sisd_insumo_ci)
matriz_sisd_ci = to_matriz(asign_pre_matriz_ci, ci = True)
matriz_hssd_ci  = pd.pivot_table(asign_pre_matriz_ci, values='valor_pond', index=['hs6_d12'], columns=['sd'], aggfunc=np.sum, fill_value=0) 


##########################################
#       EXPORTACION DE RESULTADOS        #
##########################################
asign_pre_matriz_bk.to_csv("../data/resultados/asign_pre_matriz.csv")
matriz_sisd_bk.to_csv("../data/resultados/matriz_sisd.csv")

asign_pre_matriz_ci.to_csv("../data/resultados/asign_pre_matriz_ci.csv")
matriz_sisd_ci.to_csv("../data/resultados/matriz_sisd_ci.csv")

# asign_pre_matriz_ci = pd.read_csv("../data/resultados/asign_pre_matriz_ci.csv")
# matriz_sisd_ci = pd.read_csv("../data/resultados/matriz_sisd_ci.csv")

end = datetime.datetime.now()
print(end-start)


# =============================================================================
#                       Otros
# =============================================================================

query = datos[datos["HS6"]==490199] #libros

query = datos[datos["HS6"]==640419] #zapatillas

query = datos[datos["HS6"]==842959] #topadoras
query_1 = query[query["act_ordenadas"].str.contains("K")]

query_a = datos[datos["HS6"]==842920]
query_a1 = query[query["act_ordenadas"].str.contains("K")]


#filtro para destinación de productos
# x = matriz_hssd[matriz_hssd.index.str.startswith(("870421", "870431"))]
# x.sum(axis = 0)

#impo totales (segun indec == USD 66.900 MM)
matriz_sisd_ci.sum().sum() + matriz_sisd_bk.sum().sum()


#exponentes a aplicar a la tabla de contingencia
z = pd.DataFrame(datos["HS6_d12"].value_counts()).reset_index(drop=True)
z['freq'] = z.groupby('HS6_d12')['HS6_d12'].transform('count')
z =  z.drop_duplicates()
z["expo"] = 2+np.log10(z["HS6_d12"])
sns.lineplot(data= z, x= "HS6_d12", y= "expo")


