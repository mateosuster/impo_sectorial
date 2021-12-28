
# =============================================================================
# Directorio de trabajo y librerias
# =============================================================================
import os 
# os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
# os.chdir("C:/Users/igalk/OneDrive/Documentos/laburo/CEP/procesamiento impo/nuevo1/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")

# os.getcwd()
import pandas as pd
import numpy as np
import seaborn as sns

from def_bases_nivel_ncm_12d_6act import *
from def_procesamiento_nivel_ncm_12d_6act import *
from def_matriz_nivel_ncm_12d_6act import *
from def_pre_visualizacion_nivel_ncm_12d_6act import *

#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
datos = pd.read_csv("../data/heavys/datos_clasificados_modelo_all_data.csv", sep = ";") #faltan columnas (creo q me quedó un archivo viejo)

#auxiliares
clae = pd.read_csv( "../data/clae_nombre.csv")
dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")
clae_to_ciiu = pd.read_excel("../data/Pasar de CLAE6 a CIIU3.xlsx")
hs_to_isic = pd.read_csv("../data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")
bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018_HS12.csv", sep = ";")
dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")
ncm12_desc = pd.read_csv("../data/d12_2012-2017.csv", sep=";")

vector_comercio_bk = pd.read_csv("../data/vector_de_comercio_clae_bk.csv", sep = ";").drop("Unnamed: 0", 1)
vector_comercio_ci = pd.read_csv("../data/vector_de_comercio_clae_ci.csv", sep = ";")

#############################################
#           preparación bases               #
#############################################
ncm12_desc_mod = predo_ncm12_desc(ncm12_desc )["ncm_desc"] 
letras = predo_sectores_nombres(clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae) #meter loop aca
dic_stp = predo_stp(dic_stp)
dic_propio = predo_dic_propio(clae_to_ciiu, dic_ciiu,clae)

datos = diccionario_especial(datos, dic_propio) 
# datos.to_csv("../data/heavys/importaciones_pre_intro_matriz.csv")

datos_bk , datos_bk_sin_picks, bk_picks = asignacion_stp_BK(datos, dic_stp)
datos_ci = filtro_ci(datos)


#############################################
#         BK                                #
#############################################
datos_bk_comercio = def_join_comercio(datos_bk_sin_picks , vector_comercio_bk)  #emprolijar esta funcion con un loop
tabla_contingencia = def_contingencia(datos_bk_comercio , datos) #Tabla de contingencia producto-sector   
datos_bk_comercio_pond= def_ponderaciones(datos_bk_comercio,tabla_contingencia, ci = False) #ponderación por ncm y letra
matriz_sisd_insumo = def_asignacion_sec(datos_bk_comercio_pond) ##asignación por probabilidad de G-bk (insumo para la matriz)
asign_pre_matriz= def_asignacion_prob(matriz_sisd_insumo)
matriz_sisd_bk = to_matriz(asign_pre_matriz) #matriz SISD
matriz_hssd_bk  = pd.pivot_table(asign_pre_matriz, values='valor_pond', index=['hs6_d12'], columns=['sd'], aggfunc=np.sum, fill_value=0)

# matriz_sisd_insumo.to_csv("../data/resultados/matriz_pesada_12d_6act_postML.csv", index= False)
# asign_pre_matriz.to_csv("../data/resultados/asign_pre_matriz.csv")
# matriz_sisd_bk.to_csv("../data/resultados/matriz_sisd.csv")
# matriz_hssd_bk.to_csv("../data/resultados/matriz_hssd_bk.csv")

#matriz_sisd_insumo= pd.read_csv("../data/resultados/matriz_pesada_12d_6act_postML.csv")
#matriz_sisd= pd.read_csv("../data/resultados/matriz_sisd.csv")

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

# matriz_sisd_ci= pd.read_csv("../data/resultados/matriz_sisd_ci.csv")


# =============================================================================
#                       Otros
# =============================================================================
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


# =============================================================================
#                       Visualizacion (MOVER AL OTRO SCRIPT)
# =============================================================================
#preprocesamiento
impo_tot_sec = impo_total(matriz_sisd_bk, dic_propio, sectores_desc= False)
comercio_y_propio = impo_comercio_y_propio(matriz_sisd_bk,dic_propio, sectores_desc = False)

impo_tot_sec_ci = impo_total(matriz_sisd_ci,dic_propio, sectores_desc= False)
comercio_y_propio_ci = impo_comercio_y_propio(matriz_sisd_ci,dic_propio, sectores_desc = False)

x = pd.merge(matriz_sisd_bk.reset_index(),letras_ciiu, how = "outer", left_on= "si",  right_on="letra")

# graficos
graficos(dic_propio, impo_tot_sec, comercio_y_propio, ue_dest = "bk")
graficos(dic_propio, impo_tot_sec_ci, comercio_y_propio_ci,  ue_dest= "ci")

##### tabla top 5
# Top 5 de importaciones de cada sector
top_5_impo = top_5(asign_pre_matriz, letras_ciiu , ncm12_desc_mod, impo_tot_sec) # a veces rompe por la var HS12, pero se soluciona corriendo de nuevo el preprocesamiento
top_5_impo.to_excel("../data/resultados/top5_impo.xlsx")

top_5_impo_ci = top_5(asign_pre_matriz_ci, letras_ciiu , ncm12_desc_mod, impo_tot_sec_ci) # a veces rompe por la var HS12, pero se soluciona corriendo de nuevo el preprocesamiento
top_5_impo_ci.to_excel("../data/resultados/top5_impo_ci.xlsx")


asign_pre_matriz_ci[asign_pre_matriz_ci.sd == "P"]

#CUITS QUE IMPORTAN TOP HS6 Industria
# top_industria = top_5_impo[top_5_impo["letra"]=="C"]["hs6"].iloc[[0,3]]
# cuit_top_c = join_impo_clae[join_impo_clae["HS6"].isin(top_industria )].sort_values("valor",ascending=False)#["CUIT_IMPOR"].unique()
# cuit_empresas[cuit_empresas["cuit"].isin(cuit_top_c)]




