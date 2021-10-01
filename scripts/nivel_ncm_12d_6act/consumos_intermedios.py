# -*- coding: utf-8 -*-

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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

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

dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")
# dic_stp.to_csv("../data/bsk-prod-clasificacion.csv", index=False)

ncm_sector = pd.read_csv("../nivel_clae_letra/resultados/asignacion_ncm.csv")


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
ncm_sector= predo_ncm(ncm_sector)

#############################################
#                joins                      #
#############################################
join_impo_clae = def_join_impo_clae(impo_d12, cuit_empresas)
join_impo_clae_bec_bk = def_join_impo_clae_bec(join_impo_clae, bec_bk)
# join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio)



# buscando las partes y accesorios
bec_cap = bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]
partes_accesorios  = bec[ bec["BEC5EndUse"].str.startswith("CAP", na = False) ]
partes_accesorios   =  partes_accesorios[partes_accesorios["HS6Desc"].str.contains("part|acces")]  

join_impo_clae_bec_parts = pd.merge(join_impo_clae, partes_accesorios  , how= "left" , left_on = "HS6", right_on= "HS6" )
# filtramos las impos que no mergearon (no arrancan con CAP)
join_impo_clae_bec_parts= join_impo_clae_bec_parts[join_impo_clae_bec_parts["HS4"].notnull()]
len(join_impo_clae_bec_parts) + len(join_impo_clae_bec_bk) 

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
# Reasignación STP para agro y transporte 
# =============================================================================

#arreglar transporte (camiones ) y agro (cosechadoras). revisar donde caen los de desc gral: "Maq agr y forestal"

#filtro transporte
stp_trans= dic_stp[dic_stp["utilizacion"] == "Transporte"]

# filtros agro (son iguales)
stp_agro = dic_stp[dic_stp["demanda"].str.contains("agríc", case =False)]
# stp_agro = dic_stp[dic_stp["desc_gral"].str.contains("Maquinaria agropecuaria y forestal")]
# cosechadoras y sembradoras 

# porcentaje relativo de las asignaciones
ncm_sector_perc = ncm_sector.set_index("hs6")
ncm_sector_perc = ncm_sector_perc.div(ncm_sector_perc.sum(axis=1), axis = 0).mul(100).round(3).reset_index()

# destino de los bienes de la stp 
stp_asig = pd.merge(dic_stp[["NCM", "desc", "demanda"]], ncm_sector_perc, how= "left", left_on = "NCM", right_on = "hs6" )
stp_agro_asig = pd.merge(stp_agro[["NCM", "desc", "demanda"]], ncm_sector_perc, how= "left", left_on = "NCM", right_on = "hs6" ).drop("hs6", axis=1)
stp_trans_asig = pd.merge(stp_trans[["NCM", "desc", "demanda"]], ncm_sector_perc, how= "left", left_on = "NCM", right_on = "hs6" ).drop("hs6", axis=1)

# pasar a agro donde % de C o G es mayor que el % asignado a A
# %A + %G < %C 
stp_agro_filtro =stp_agro_asig[stp_agro_asig["A"]+stp_agro_asig["G"] < stp_agro_asig["C"] ]
# filtro1 = join_impo_clae_bec_bk[~join_impo_clae_bec_bk["HS6"].isin( pd.concat([stp_agro_filtro["NCM"] , stp_trans["NCM"]]))]
# join_impo_clae_bec_bk["ue_dest"] = np.where(join_impo_clae_bec_bk[join_impo_clae_bec_bk["HS6"].isin( pd.concat([stp_agro_filtro["NCM"] , stp_trans["NCM"]]))])

# =============================================================================
# Antes de empezar... feature destinación limpio
# ============================================================================= 

join_impo_clae_bec_bk["dest_clean"] = join_impo_clae_bec_bk["destinacion"].apply(lambda x: destinacion_limpio(x))
join_impo_clae_bec_bk["dest_clean"].value_counts()
114256 - join_impo_clae_bec_bk["dest_clean"].value_counts().sum()


# =============================================================================
# 1er filtro: STP entre específicos y generales
# =============================================================================
#frecuencia 
dic_stp["utilizacion"].value_counts()

# vectores filtros
# stp_general = dic_stp[dic_stp["utilizacion"]=="General"] # =~ CI
stp_especifico = dic_stp[dic_stp["utilizacion"]=="Específico"] # =~ BK

join_impo_clae_bec_bk["ue_dest"] = np.where(join_impo_clae_bec_bk["HS6"].isin(stp_especifico ["NCM"]), 
                                            np.where( ~join_impo_clae_bec_bk["destinacion"].str.contains("PARA TRANSF|C/TRANS|P/TRANS|RAF|C/TRNSF|ING.ZF INSUMOS", 
                                            # np.where( join_impo_clae_bec_bk["destinacion"].str.contains("S/TRAN|SIN TRANSF|INGR.ZF BIENES", 
                                                                                                         case=False), 
                                                     "BK", ""
                                                     ), ""
                                            )
                                            
join_impo_clae_bec_bk["ue_dest"].value_counts()

# filtrado 1
filtro1 = join_impo_clae_bec_bk[join_impo_clae_bec_bk["ue_dest"]==""]


# detección partes y piezas vía text mining
# partes_stp = stp_asig[stp_asig["desc"].str.contains("parte", case = False) ]
# accesorios_stp = stp_asig[stp_asig["desc"].str.contains("accesorio", case = False) ]

# =============================================================================
# 2do filtro: Variable Destinacion
# =============================================================================
# notas 
# A) IT S/TR = BK (n dest = 1)
# B) IT C/TR = CI (n dest = 1)
# C) IT CONS (n dest = 1) ///  IC  ("a secas") => Clasf supervisada 
# D) IT S/TR &  ICons  => sobre IC aplicar 
# E) IT C/TR &  IConc  => sobre IC aplicar diferencia de media de la distribución vs observacion) (o alejamiento 3 desvios) (o MAD !)
# F) C/TR & S/TR 
# G) IT C/TR & IT S/TR &  ICons  => Clasf supervisada 

# A= BK
# B = CI
# D --> Zscore mod. si mayor a 3.5veces = BK
# F --> D -->  Zscore mod. si menor a 3.5veces = CI
# G --> clasificación supervisada (si se puede 🙏)
# C --> Con todas las etiquetas previas puestas. Clasificación supervisada (si se puede 🙏)


##### Filtros en partidas con una unica destinacion
filtro1["destinacion"].value_counts()#.sum()
filtro1["dest_clean"].value_counts()#.sum()

# NUEVA PROPUESTA
# partidas_n_dest = join_impo_clae_bec_bk.groupby(["HS6_d12", "destinacion"],as_index = False).size().loc[lambda x: ~x["HS6_d12"].isin(partidas_dest["HS6_d12"])]
ya_filtrado = join_impo_clae_bec_bk[join_impo_clae_bec_bk["ue_dest"]!=""]

(len(filtro1) + len(ya_filtrado)) == len(join_impo_clae_bec_bk)

filtro1st =  filtro1[filtro1["dest_clean"] == "S/TR"] 
filtro1ct =  filtro1[filtro1["dest_clean"] == "C/TR"] 
filtro1co =  filtro1[filtro1["dest_clean"] == "CONS&Otros"] 

len(filtro1)  == ( len(filtro1st) +len(filtro1ct) + len(filtro1co) ) 

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

dest_a = filtro1[filtro1["HS6_d12"].isin( filtro_a  )]
dest_b = filtro1[filtro1["HS6_d12"].isin( filtro_b  )]
dest_c = filtro1[filtro1["HS6_d12"].isin( filtro_c  )]
dest_d = filtro1[filtro1["HS6_d12"].isin( filtro_d  )]
dest_e = filtro1[filtro1["HS6_d12"].isin( filtro_e  )]
dest_f = filtro1[filtro1["HS6_d12"].isin( filtro_f  )]
dest_g = filtro1[filtro1["HS6_d12"].isin( filtro_g )]

dest_a["filtro"] = "A" 
dest_b["filtro"] = "B" 
dest_c["filtro"] = "C" 
dest_d["filtro"] = "D" 
dest_e["filtro"] = "E"                                          
dest_f["filtro"] = "F"
dest_g["filtro"] = "G"  
       

# =============================================================================
# Consistencia filtro 2
# =============================================================================

#importaciones detectadas con S/T, C/T y Consumo
len(filtro1st) +len(filtro1ct) + len(filtro1co) 

(len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) +len(dest_b)+ len(dest_c)) ==len(filtro1)

# Concateno el filtro 2
filtro1 = pd.concat( [dest_a, dest_b, dest_c, dest_d,  dest_e,  dest_f, dest_g], axis = 0)


filtro1["ue_dest"].value_counts()
filtro1["ue_dest"] = np.where((filtro1["filtro"] == "A") & 
                                (filtro1["dest_clean"] == "S/TR"), 
                                    "BK",  
                                    np.where( (filtro1["filtro"] == "B") & 
                                            (filtro1["dest_clean"] == "C/TR"),
                                            "CI", ""
                                            ) 
                                ) 

filtro2 = filtro1[filtro1["ue_dest"] == "" ]
len(filtro2)/len(filtro1)-1

filtro2["filtro"].value_counts()
filtro2[["dest_clean", "filtro"]].value_counts()
filtro2[["dest_clean","filtro", "uni_decl" ]].value_counts()
filtro2[["dest_clean","filtro", "uni_est" ]].value_counts()



# =============================================================================
# ## Aplicando métrica
# =============================================================================
# a. el bien 1234567 fue comprado con dos destinaciones: s/tr y cons. y con diferentes unidades de medida:
# b. dentro de cada unidad de medida a s/tr le llamamos BK. calculamos su mediana, y hacemos zscore sobre los "cons". 
# c. si calculamos zscore sobre cantidades, aquellos que se alejan 3.5 del zscrore por arriba los llamamos CI. (es menos probable que algo que se compra mucho se un bK)
# si calculamos zscore sobre precios o kg, aquellos que se alejan 3.5 del zscrore por abajo los llamamos CI. (es menos probable que algo liviano o barato sea bk)

from scipy.stats import median_abs_deviation, zscore

def metrica(x):
    return (x["valor"] * x["kilos"])/x["cant_decl"]

def mod_z(col: pd.Series, thresh: float=3.5):
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = 0.645* ((col - med_col) / med_abs_dev)
    # mod_z = mod_z[np.abs(mod_z) < thresh]
    return np.abs(mod_z)

# Aplico métrica 1
# dest_c["metric"] = dest_c.apply(lambda x : (x["valor"] * x["kilos"])/x["cant_decl"] , axis = 1)
filtro2["metric"] = filtro2.apply(lambda x : metrica(x), axis = 1)

# CASO D
#con uni decl
# filtro2_centerD = filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
# filtro2_D = pd.merge(filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_centerD.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

#con uni est
filtro2_D_mad= filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
filtro2_D_median = filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
filtro2_D = pd.merge(filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_D_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_est", "filtro"], right_on = ["HS6_d12", "uni_est", "filtro"])
filtro2_D = pd.merge(filtro2_D, filtro2_D_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_est", "filtro"], right_on = ["HS6_d12", "uni_est", "filtro"])

filtro2_D["mad"] = np.where(filtro2_D["mad"]==0, 0.001, filtro2_D["mad"]  )
filtro2_D["brecha"] = (filtro2_D["metric"]/filtro2_D["median"])-1
filtro2_D["z_score"] = filtro2_D.apply(lambda x: 0.6745*((x["metric"]-x["median"]))/(x["mad"]), axis =1 )
filtro2_D["ue_dest"] = np.where(filtro2_D["z_score"] <= -3.5, "CI","BK" ) # no conviene establecer los límites al reves? con un -3.5??

filtro2_D_bk = filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ]
filtro2_D_bk["ue_dest"] = "BK"

clasif_D = pd.concat([filtro2_D_bk, filtro2_D.drop(["median", "mad", "brecha", "z_score"], axis = 1)] )
clasif_D["ue_dest"].value_counts()
clasif_D.groupby(["ue_dest"])["valor"].sum()

# CASO E
#con uni est
filtro2_E_mad= filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
filtro2_E_median = filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
filtro2_E = pd.merge(filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_E_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_est", "filtro"], right_on = ["HS6_d12", "uni_est", "filtro"])
filtro2_E = pd.merge(filtro2_E, filtro2_E_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_est", "filtro"], right_on = ["HS6_d12", "uni_est", "filtro"])

filtro2_E["mad"] = np.where(filtro2_E["mad"]==0, 0.001, filtro2_E["mad"]  )
filtro2_E["brecha"] = (filtro2_E["metric"]/filtro2_E["median"])-1
filtro2_E["z_score"] = filtro2_E.apply(lambda x: 0.6745*((x["metric"]-x["median"]))/(x["mad"]), axis =1 )
filtro2_E["ue_dest"] = np.where(filtro2_E["z_score"] <= 3.5, "CI","BK" )

filtro2_E_ci = filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="C/TR") ]
filtro2_E_ci["ue_dest"] = "CI"

clasif_E = pd.concat([filtro2_E_ci, filtro2_E.drop(["median", "mad", "brecha", "z_score"], axis = 1) ]) 
clasif_E["ue_dest"].value_counts()


# CASO G
# filtro2[filtro2["filtro"]=="G"].value_counts("dest_clean")
filtro2_G_mad_bk = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12" , "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_bk"}).reset_index(drop = True)
filtro2_G_median_bk = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_bk"}).reset_index(drop = True)
filtro2_G_mad_ci = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_ci"}).reset_index(drop = True)
filtro2_G_median_ci = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_ci"}).reset_index(drop = True)

filtro2_G = pd.merge(filtro2[(filtro2["filtro"]=="G")  & (filtro2["dest_clean"]=="CONS&Otros")] , filtro2_G_mad_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left", left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
filtro2_G = pd.merge(filtro2_G , filtro2_G_median_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
filtro2_G = pd.merge(filtro2_G , filtro2_G_mad_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
filtro2_G = pd.merge(filtro2_G , filtro2_G_median_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])

filtro2_G ["mad_bk"] = np.where(filtro2_G ["mad_bk"]==0, 0.001, filtro2_G ["mad_bk"]  )
filtro2_G ["mad_ci"] = np.where(filtro2_G ["mad_ci"]==0, 0.001, filtro2_G ["mad_ci"]  )
# filtro2_G ["brecha"] = (filtro2_G ["metric"]/filtro2_G ["median"])-1
filtro2_G ["z_score_bk"] = filtro2_G .apply(lambda x: 0.6745*((x["metric"]-x["median_bk"]))/(x["mad_bk"]), axis =1 )
filtro2_G ["z_score_ci"] = filtro2_G .apply(lambda x: 0.6745*((x["metric"]-x["median_ci"]))/(x["mad_ci"]), axis =1 )

filtro2_G["ue_dest"] = np.where(np.abs(filtro2_G["z_score_ci"]) > np.abs(filtro2_G["z_score_bk"]), "BK", "CI" ) 
filtro2_G["ue_dest"].value_counts() 

filtro2_G_clas = filtro2[(filtro2["filtro"]=="G")  & (filtro2["dest_clean"]!="CONS&Otros")]
filtro2_G_clas ["ue_dest"] = np.where(filtro2_G_clas["dest_clean"]== "S/TR","BK", "CI" ) 
filtro2_G_clas["ue_dest"].value_counts()

clasif_G = pd.concat([filtro2_G_clas, filtro2_G.drop(["median_bk", "median_ci",  "mad_bk", "mad_ci", "z_score_bk", "z_score_ci"],axis =1 )], axis = 0) 
clasif_G ["ue_dest"].value_counts()


## EDA de Datos ya clasificados
clasif_AB = filtro1[filtro1["ue_dest"] != "" ]
clasif_AB["metric"] = clasif_AB.apply(lambda x: metrica(x), axis = 1)

data_clasif = pd.concat([clasif_AB, clasif_D ,clasif_E,clasif_G  ], axis= 0)
data_clasif["ue_dest"].value_counts()

# data_clasif["metric_zscore"] = (data_clasif["metric"] -data_clasif["metric"].mean())/ data_clasif["metric"].std(ddof=1)  


# =============================================================================
#  Exportacion de datos clasificados con UE dest
# =============================================================================
## DATOS CLASIFICADOS
stp_ue_dest = join_impo_clae_bec_bk[join_impo_clae_bec_bk["ue_dest"]=="BK" ]
stp_ue_dest ["ue_dest"].value_counts()
stp_ue_dest ["metric"] = stp_ue_dest .apply(lambda x: metrica(x), axis = 1)
stp_ue_dest["filtro"] = "STP"

data_clasif_ue_dest = pd.concat([data_clasif, stp_ue_dest], axis = 0)
data_clasif_ue_dest ["ue_dest"].value_counts()

data_clasif_ue_dest["precio_kilo"]= data_clasif_ue_dest["valor"]/data_clasif_ue_dest["kilos"]
data_clasif_ue_dest.to_csv("../data/resultados/bk_con_ue_dest.csv")


## DATOS NO CLASIFICADOS
data_not_clasif = pd.concat( [ dest_c], axis = 0)
data_not_clasif["ue_dest"] = pd.np.NaN #np.NAN

# data_not_clasif.isna().sum()
data_not_clasif["metric"] = data_not_clasif.apply(lambda x: metrica(x), axis = 1)
data_not_clasif["precio_kilo"]= data_not_clasif["valor"]/data_not_clasif["kilos"]

data_not_clasif.to_csv("../data/resultados/bk_sin_ue_dest.csv")


# data_clasif.to_csv("data_clasif.csv")
# data_not_clasif.to_csv("data_not_clasif.csv")
# set(data_clasif.columns).symmetric_difference(set(data_not_clasif.columns))






################
# algunos boxplots
ax = sns.boxplot(x= "ue_dest", y = "metric_zscore", hue="ue_dest" , data = data_clasif)
ax = sns.boxplot(x= "ue_dest", y = "metric", hue="ue_dest" , data = data_clasif)
ax.set(ylabel = "métrica en escala logarítmica")
ax.set_yscale("log")

ax = sns.boxplot(x= "ue_dest", y = "metric", hue="ue_dest" , data = data_clasif, showfliers = False)
ax = sns.boxplot(x= "ue_dest", y = "kilos", hue="ue_dest" , data = data_clasif, showfliers = False)

ax = sns.boxplot(x= "ue_dest", y = "precio_kilo", hue="ue_dest" , data = data_clasif, showfliers = False)
###############



## ALL DATA

all_data = pd.concat([data_clasif, data_not_clasif], axis = 0)
all_data["precio_kilo"]=all_data["valor"]/all_data["kilos"]
all_data["HS6"] = all_data["HS6_d12"].str.slice(0,6).astype("category")
all_data["HS6_d12"] = all_data["HS6_d12"].astype("category")
all_data["ue_dest"] = all_data["ue_dest"].astype("category")

all_data = all_data [["HS6_d12", "HS6", "ue_dest", 'valor',  'kilos', "precio_kilo" ,'cant_est', "metric" ]]
# all_data .info()
all_data.isna().sum()

# df_na = all_data[all_data["ue_dest"].isna()]
# df_com = all_data[~all_data["ue_dest"].isna()]


############################# KNN
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

cat_variables = all_data[["HS6_d12", "HS6"]]
cat_dummies = pd.get_dummies(cat_variables, drop_first=True)

all_data["bk_dummy"] = all_data["ue_dest"].map({"BK": 1, "CI": 0})
all_data["bk_dummy"].isna().sum()

data_2fill = all_data.drop(["HS6_d12", "HS6", "ue_dest"], axis=1)
data_2fill= pd.concat([data_2fill, cat_dummies], axis=1)

imputer = KNNImputer(n_neighbors=2)
df_fill = pd.DataFrame(imputer.fit_transform(data_2fill),columns = data_2fill.columns)



# =============================================================================
# HASTA ACA ESTA LIMPIOOOOOOOO
# =============================================================================
#####
scaler = MinMaxScaler()
df = df.drop("ue_dest", axis =1)
df = pd.DataFrame(scaler.fit_transform(df ), columns = df .columns)

df.append(all_data["ue_dest"], ignore_index=True)

data = pd.concat([df.reset_index(drop=True), pd.DataFrame(all_data["ue_dest"]).reset_index(drop=True) ], axis=1)


################## 
### imputacion con depp leearning

import datawig

df_train, df_test = datawig.utils.random_split(train)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=["HS6_d12", "HS6", 'valor',  'kilos', "precio_kilo" ,'cant_est', "metric" ], # column(s) containing information about the column we want to impute
    output_column= "ue_dest", # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=50)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)



###################
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")
imp.fit(all_data)
data_complete = imp.transform(all_data)
data_complete  = pd.DataFrame(data_complete )

data_complete.columns()

# =============================================1================================
# 4to filtro: Metricas 
# =============================================================================
# probar con cantidad importada (en sus respectivas unidades declaradas y en kilos, y en peso unitario)


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



# np.array([x, label])

# type(label)
# type(x)
# len(label)
# type(x)
# label.shape
# x.reshape(len(label))

# x = kmeans.fit(data[data["groupID"]== filtro[0]]["cant_decl"])



# combinacion de unidades 
# x = impo_d12.groupby(["uni_est", "uni_decl"]).size().reset_index().rename(columns={0:'count'}).sort_values("count", ascending = False)
# x["frec_relativa"] = x["count"]/(x["count"].sum() )

# x = impo_d12[impo_d12["HS6_d12"].isin(n_emp_1["HS6_d12"])].groupby(["unidad_est", "unidad_decl"]).size().reset_index().rename(columns={0:'count'}).sort_values("count", ascending = False)
# x["frec_relativa"] = x["count"]/(x["count"].sum() )


# y = impo_d12[(impo_d12["HS6_d12"].isin(n_emp_1["HS6_d12"])) & (impo_d12["unidad_est"]=="Unidad") & (impo_d12["unidad_decl"]=="Unidad") ]
# # y = impo_d12[(impo_d12["HS6_d12"].isin(n_emp_1["HS6_d12"])) ]
# y.valor.sum() / impo_d12.valor.sum()


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
