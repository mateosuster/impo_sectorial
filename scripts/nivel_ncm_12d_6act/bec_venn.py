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
# import plotinpy as pnp

from scipy.stats import median_abs_deviation , zscore

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
# impo_d12 = pd.read_csv("../data/IMPO_17_feature.csv")
# impo_d12 = pd.read_csv("../data/IMPO_2017_12d.csv")
#impo_17 = pd.read_csv(  "../data/IMPO_2017.csv", sep=";")
impo_d12 = pd.read_csv("../data/impo_2017_diaria.csv")      # impo DIARIA
impo_d12["ANYO"] = impo_d12["FECH_OFIC"].str.slice(0,4) 


clae = pd.read_csv( "../data/clae_nombre.csv")
comercio = pd.read_csv("../data/comercio_clae.csv", encoding="latin1")
#cuit_clae = pd.read_csv( "../data/cuit 2017 impo_con_actividad.csv")
cuit_clae = pd.read_csv( "../data/Cuit_todas_las_actividades.csv")
bec = pd.read_csv( "../data/HS2012-17-BEC5 -- 08 Nov 2018_HS12.csv", sep = ";")
# bec_to_clae = pd.read_csv("../data/bec_to_clae.csv")
dic_stp = pd.read_excel("../data/bsk-prod-clasificacion.xlsx")

#diccionario ncm12d
# ncm12_desc = pd.read_csv("../data/NCM 12d.csv", sep=";")
# ncm12_desc_split = pd.concat([ncm12_desc.iloc[:,0], pd.DataFrame(ncm12_desc['Descripción Completa'].str.split('//', expand=True))], axis=1)

ncm12_desc = pd.read_csv("../data/d12_2012-2017.csv", sep=";")
ncm12_desc = ncm12_desc[["POSICION", "DESCRIPCIO"]]
ncm12_desc.rename(columns = {"POSICION": "Posición", "DESCRIPCIO":"Descripción Completa"}, inplace = True)
ncm12_desc_split = pd.concat([ncm12_desc.iloc[:,0], pd.DataFrame(ncm12_desc['Descripción Completa'].str.split('//', expand=True))], axis=1)

# parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")

# bce_cambiario = pd.read_csv("../data/balance_cambiario.csv", skiprows = 3, error_bad_lines=False, sep= ";", na_values =['-'])
# isic = pd.read_csv("../data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )
# dic_ciiu = pd.read_excel("../data/Diccionario CIIU3.xlsx")



#############################################
#           preparación bases               #
#############################################

# predo_impo_17(impo_17)
impo_d12  = predo_impo_12d(impo_d12, ncm12_desc)
letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)
cuit_empresas= predo_cuit_clae(cuit_clae, clae)
bec_bk = predo_bec_bk(bec)#, bec_to_clae)
dic_stp = predo_stp(dic_stp )

#############################################
#                joins                      #
#############################################

join_impo_clae = def_join_impo_clae(impo_d12, cuit_empresas)
join_impo_clae_bec_bk = def_join_impo_clae_bec(join_impo_clae, bec_bk)
join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio)


# =============================================================================
# EDA BEC5
# =============================================================================
join_impo_clae["dest_clean"] = join_impo_clae["destinacion"].apply(lambda x: destinacion_limpio(x))
join_impo_clae["dest_clean"].value_counts()

impo_bec = pd.merge(join_impo_clae, bec[["HS6", "BEC5EndUse" ]], how= "left" , left_on = "HS6", right_on= "HS6" )
impo_bec[impo_bec["BEC5EndUse"].isnull()]
impo_d12[impo_d12["descripcion"].isnull()]


bec["BEC5EndUse"].value_counts().sum()
bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]["BEC5EndUse"].value_counts()#.sum()
bec[bec["BEC5EndUse"].str.startswith("INT", na = False)]["BEC5EndUse"].value_counts()#.sum()
bec[bec["BEC5EndUse"].str.startswith("CONS", na = False)]["BEC5EndUse"].value_counts()#.sum()

# =============================================================================
# Bienes de capital
# =============================================================================
# =============================================================================
#  FILTRO STP entre específicos y generales
# =============================================================================
dic_stp["utilizacion"].value_counts()

# vectores filtros
# stp_general = dic_stp[dic_stp["utilizacion"]=="General"] # =~ CI
stp_especifico = dic_stp[dic_stp["utilizacion"]=="Específico"] # =~ BK


join_impo_clae_bec_bk["dest_clean"] = join_impo_clae_bec_bk["destinacion"].apply(lambda x: destinacion_limpio(x))

# opción 1
join_impo_clae_bec_bk["ue_dest"] = np.where(join_impo_clae_bec_bk["HS6"].isin(stp_especifico ["NCM"]), 
                                            np.where( ~join_impo_clae_bec_bk["destinacion"].str.contains("PARA TRANSF|C/TRANS|P/TRANS|RAF|C/TRNSF|ING.ZF INSUMOS", 
                                            # np.where( join_impo_clae_bec_bk["destinacion"].str.contains("S/TRAN|SIN TRANSF|INGR.ZF BIENES", 
                                                                                                         case=False), 
                                                     "BK", ""), "" )
                                            
join_impo_clae_bec_bk["ue_dest"].value_counts()

# opción 2
# join_impo_clae["ue_dest"] = np.where(join_impo_clae["HS6"].isin(stp_especifico ["NCM"]), 
#                                             np.where( ~join_impo_clae["destinacion"].str.contains("PARA TRANSF|C/TRANS|P/TRANS|RAF|C/TRNSF|ING.ZF INSUMOS", 
#                                             # np.where( join_impo_clae["destinacion"].str.contains("S/TRAN|SIN TRANSF|INGR.ZF BIENES", 
#                                                                                                          case=False), 
#                                                      "BK", ""), "")
                                            
# join_impo_clae["ue_dest"].value_counts()


# filtrado 1
filtro1 = join_impo_clae_bec_bk[join_impo_clae_bec_bk["ue_dest"]==""]


# =============================================================================
# FILTRO DESTINACION
# =============================================================================

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
       

# Consistencia filtro 2
#importaciones detectadas con S/T, C/T y Consumo
len(filtro1st) +len(filtro1ct) + len(filtro1co) 

(len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) +len(dest_b)+ len(dest_c)) ==len(filtro1)

# Concateno el filtro 2
filtro1 = pd.concat( [dest_a, dest_b, dest_c, dest_d,  dest_e,  dest_f, dest_g], axis = 0)


# filtro1["ue_dest"].value_counts()
filtro1["ue_dest"] = np.where((filtro1["filtro"] == "A") & 
                                (filtro1["dest_clean"] == "S/TR"), 
                                    "BK",  
                                    np.where( (filtro1["filtro"] == "B") & 
                                            (filtro1["dest_clean"] == "C/TR"),
                                            "CI", ""
                                            ) 
                                ) 

filtro1[["ue_dest", "filtro"]].value_counts()


filtro2 = filtro1[filtro1["ue_dest"] == "" ]
# len(filtro2)/len(filtro1)-1

# filtro2["filtro"].value_counts()
# filtro2[["dest_clean", "filtro"]].value_counts()
# filtro2[["dest_clean","filtro", "uni_decl" ]].value_counts()
# filtro2[["dest_clean","filtro", "uni_est" ]].value_counts()



# =============================================================================
# ## Aplicando métrica
# =============================================================================

# Aplico métrica 1
# dest_c["metric"] = dest_c.apply(lambda x : (x["valor"] * x["kilos"])/x["cant_decl"] , axis = 1)
filtro2["metric"] = filtro2.apply(lambda x : metrica(x), axis = 1)

# CASO D
#con uni decl
# filtro2_centerD = filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
# filtro2_D = pd.merge(filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_centerD.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

#con uni est
filtro2_D_mad= filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
filtro2_D_median = filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
filtro2_D = pd.merge(filtro2[(filtro2["filtro"]=="D") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_D_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])
filtro2_D = pd.merge(filtro2_D, filtro2_D_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

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
filtro2_E_mad= filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
filtro2_E_median = filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
filtro2_E = pd.merge(filtro2[(filtro2["filtro"]=="E") & (filtro2["dest_clean"]=="CONS&Otros") ] , filtro2_E_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])
filtro2_E = pd.merge(filtro2_E, filtro2_E_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

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
filtro2_G_mad_bk = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12" , "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_bk"}).reset_index(drop = True)
filtro2_G_median_bk = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_bk"}).reset_index(drop = True)
filtro2_G_mad_ci = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_ci"}).reset_index(drop = True)
filtro2_G_median_ci = filtro2[(filtro2["filtro"]=="G") & (filtro2["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_ci"}).reset_index(drop = True)

filtro2_G = pd.merge(filtro2[(filtro2["filtro"]=="G")  & (filtro2["dest_clean"]=="CONS&Otros")] , filtro2_G_mad_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left", left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
filtro2_G = pd.merge(filtro2_G , filtro2_G_median_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
filtro2_G = pd.merge(filtro2_G , filtro2_G_mad_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
filtro2_G = pd.merge(filtro2_G , filtro2_G_median_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])

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
# data_clasif["metric_zscore"] = (data_clasif["metric"] -data_clasif["metric"].mean())/ data_clasif["metric"].std(ddof=1)  

data_clasif[["filtro", "ue_dest"]].value_counts()

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
# data_clasif_ue_dest.to_csv("../data/resultados/bk_con_ue_dest.csv")


## DATOS NO CLASIFICADOS
data_not_clasif = pd.concat( [ dest_c], axis = 0)
data_not_clasif["ue_dest"] = "?" #np.NAN

# data_not_clasif.isna().sum()
data_not_clasif["metric"] = data_not_clasif.apply(lambda x: metrica(x), axis = 1)
data_not_clasif["precio_kilo"]= data_not_clasif["valor"]/data_not_clasif["kilos"]

# data_not_clasif.to_csv("../data/resultados/bk_sin_ue_dest.csv")

len(data_not_clasif) + len(data_clasif_ue_dest)



# =============================================================================
# VENN BK (VIEJO)
# =============================================================================
# impo_bec_bk = impo_bec[impo_bec["BEC5EndUse"].str.startswith("CAP", na = False)] 
# impo_bec_bk ["dest_clean"].value_counts()#.sum()
# len(impo_bec_bk )

# filtro1st =  impo_bec_bk [impo_bec_bk ["dest_clean"] == "S/TR"] 
# filtro1ct =  impo_bec_bk [impo_bec_bk ["dest_clean"] == "C/TR"] 
# filtro1co =  impo_bec_bk [impo_bec_bk ["dest_clean"] == "CONS&Otros"] 

# len(impo_bec_bk )  == ( len(filtro1st) +len(filtro1ct) + len(filtro1co) ) 

# # Filtros de conjuntos
# set_st = set(filtro1st["HS6_d12"])
# set_ct = set(filtro1ct["HS6_d12"])
# set_co = set(filtro1co["HS6_d12"])

# filtro_a = set_st - set_co - set_ct 
# filtro_b = set_ct - set_st - set_co 
# filtro_c = set_co - set_ct -set_st 

# filtro_d = (set_st & set_co) - (set_ct )
# filtro_e = (set_ct & set_co) - ( set_st)
# filtro_f = (set_ct & set_st) - set_co 
# filtro_g = set_ct & set_st & set_co 

# dest_a = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_a  )]
# dest_b = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_b  )]
# dest_c = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_c  )]
# dest_d = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_d  )]
# dest_e = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_e  )]
# dest_f = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_f  )]
# dest_g = impo_bec_bk[impo_bec_bk["HS6_d12"].isin( filtro_g )]

# (len(dest_d) + len(dest_e) + len(dest_f) + len(dest_g) + len(dest_a) +len(dest_b)+ len(dest_c)) ==len(impo_bec_bk)

# dest_a["filtro"] = "A" 
# dest_b["filtro"] = "B" 
# dest_c["filtro"] = "C" 
# dest_d["filtro"] = "D" 
# dest_e["filtro"] = "E"                                          
# dest_f["filtro"] = "F"
# dest_g["filtro"] = "G"  

# bk = pd.concat( [dest_a, dest_b, dest_c, dest_d,  dest_e,  dest_f, dest_g], axis = 0)


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

dest_a["filtro"] = "A" 
dest_b["filtro"] = "B" 
dest_c["filtro"] = "C" 
dest_d["filtro"] = "D" 
dest_e["filtro"] = "E"                                          
dest_f["filtro"] = "F"
dest_g["filtro"] = "G"  

cons_int = pd.concat( [dest_a, dest_b, dest_c, dest_d,  dest_e,  dest_f, dest_g], axis = 0)
cons_int["metric"] = cons_int.apply(lambda x : metrica(x), axis = 1)

# A , B, C, E
cons_int["ue_dest"] = np.where((cons_int["filtro"] == "B") |
                               (cons_int["filtro"] == "E") |
                               (cons_int["filtro"] == "C") , "CI",  
                                np.where((cons_int["filtro"] == "A"), "BK",
                                        np.nan) )
# D
cons_int_D_mad= cons_int[(cons_int["filtro"]=="D") & (cons_int["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
cons_int_D_median = cons_int[(cons_int["filtro"]=="D") & (cons_int["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
cons_int_D = pd.merge(cons_int[(cons_int["filtro"]=="D") & (cons_int["dest_clean"]=="CONS&Otros") ] , cons_int_D_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])
cons_int_D = pd.merge(cons_int_D, cons_int_D_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

cons_int_D["mad"] = np.where(cons_int_D["mad"]==0, 0.001, cons_int_D["mad"]  )
cons_int_D["z_score"] = cons_int_D.apply(lambda x: 0.6745*((x["metric"]-x["median"]))/(x["mad"]), axis =1 )
cons_int_D["ue_dest"] = np.where(cons_int_D["z_score"] > 3.5, "BK", "CI" ) 

cons_int_D_st = cons_int[(cons_int["filtro"]=="D") & (cons_int["dest_clean"]=="S/TR") ]
cons_int_D_st["ue_dest"] = "BK"

cons_int_D = pd.concat([cons_int_D, cons_int_D_st], axis = 0)
# cons_int_D["ue_dest"].value_counts()

# CASO G
# cons_int[cons_int["filtro"]=="G"].value_counts("dest_clean")
cons_int_G_mad_bk = cons_int[(cons_int["filtro"]=="G") & (cons_int["dest_clean"]=="S/TR") ].groupby(["HS6_d12" , "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_bk"}).reset_index(drop = True)
cons_int_G_median_bk = cons_int[(cons_int["filtro"]=="G") & (cons_int["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_bk"}).reset_index(drop = True)
cons_int_G_mad_ci = cons_int[(cons_int["filtro"]=="G") & (cons_int["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_ci"}).reset_index(drop = True)
cons_int_G_median_ci = cons_int[(cons_int["filtro"]=="G") & (cons_int["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_ci"}).reset_index(drop = True)

cons_int_G = pd.merge(cons_int[(cons_int["filtro"]=="G")  & (cons_int["dest_clean"]=="CONS&Otros")] , cons_int_G_mad_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left", left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
cons_int_G = pd.merge(cons_int_G , cons_int_G_median_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
cons_int_G = pd.merge(cons_int_G , cons_int_G_mad_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
cons_int_G = pd.merge(cons_int_G , cons_int_G_median_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])

cons_int_G ["mad_bk"] = np.where(cons_int_G ["mad_bk"]==0, 0.001, cons_int_G ["mad_bk"]  )
cons_int_G ["mad_ci"] = np.where(cons_int_G ["mad_ci"]==0, 0.001, cons_int_G ["mad_ci"]  )
cons_int_G ["z_score_bk"] = cons_int_G .apply(lambda x: 0.6745*((x["metric"]-x["median_bk"]))/(x["mad_bk"]), axis =1 )
cons_int_G ["z_score_ci"] = cons_int_G .apply(lambda x: 0.6745*((x["metric"]-x["median_ci"]))/(x["mad_ci"]), axis =1 )

cons_int_G["ue_dest"] = np.where(np.abs(cons_int_G["z_score_ci"]) > np.abs(cons_int_G["z_score_bk"]), "BK", "CI" ) 
cons_int_G["ue_dest"].value_counts() 

cons_int_G_no_cons = cons_int[(cons_int["filtro"]=="G")  & (cons_int["dest_clean"]!="CONS&Otros")]
cons_int_G_no_cons["ue_dest"] = np.where(cons_int_G_no_cons["dest_clean"]== "S/TR","BK", "CI" ) 
# cons_int_G_clas["ue_dest"].value_counts()

cons_int_G_clasif = pd.concat([cons_int_G_no_cons, cons_int_G.drop(["median_bk", "median_ci",  "mad_bk", "mad_ci", "z_score_bk", "z_score_ci"],axis =1 )], axis = 0) 
cons_int_G_clasif["ue_dest"].value_counts()#.sum()

cons_int_clasif =pd.concat([cons_int_D, cons_int_G_clasif, cons_int[cons_int["filtro"].str.contains("A|B|C|E")]  ])
cons_int_clasif ["ue_dest"].value_counts()#.sum()
cons_int_clasif [["filtro", "ue_dest"]].value_counts()#.sum()



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

dest_a["filtro"] = "A" 
dest_b["filtro"] = "B" 
dest_c["filtro"] = "C" 
dest_d["filtro"] = "D" 
dest_e["filtro"] = "E"                                          
dest_f["filtro"] = "F"
dest_g["filtro"] = "G"  

cons_fin = pd.concat( [dest_a, dest_b, dest_c, dest_d,  dest_e,  dest_f, dest_g], axis = 0)

# A, B, C, E
cons_fin["ue_dest"] = np.where((cons_fin["filtro"] == "B") |
                               (cons_fin["filtro"] == "E") |
                               (cons_fin["filtro"] == "C") , "CI",  
                               np.where((cons_fin["filtro"] == "A"), "BK",
                                        np.nan)  )
cons_fin["ue_dest"].value_counts()


cons_fin["metric"] = cons_fin.apply(lambda x : metrica(x), axis = 1)

#D
#con uni est
cons_fin_D_mad= cons_fin[(cons_fin["filtro"]=="D") & (cons_fin["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
cons_fin_D_median = cons_fin[(cons_fin["filtro"]=="D") & (cons_fin["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
cons_fin_D = pd.merge(cons_fin[(cons_fin["filtro"]=="D") & (cons_fin["dest_clean"]=="CONS&Otros") ] , cons_fin_D_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])
cons_fin_D = pd.merge(cons_fin_D, cons_fin_D_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_decl", "filtro"], right_on = ["HS6_d12", "uni_decl", "filtro"])

cons_fin_D["mad"] = np.where(cons_fin_D["mad"]==0, 0.001, cons_fin_D["mad"]  )
cons_fin_D["brecha"] = (cons_fin_D["metric"]/cons_fin_D["median"])-1
cons_fin_D["z_score"] = cons_fin_D.apply(lambda x: 0.6745*((x["metric"]-x["median"]))/(x["mad"]), axis =1 )
cons_fin_D["ue_dest"] = np.where(cons_fin_D["z_score"] > 3.5, "BK", "CI" ) 

cons_fin_D_st = cons_fin[(cons_fin["filtro"]=="D") & (cons_fin["dest_clean"]=="S/TR") ]
cons_fin_D_st["ue_dest"] = "BK"

cons_fin_D = pd.concat([cons_fin_D, cons_fin_D_st], axis = 0)
# cons_fin_D["ue_dest"].value_counts()#.sum()


# CASO G
# cons_fin[cons_fin["filtro"]=="G"].value_counts("dest_clean")
cons_fin_G_mad_bk = cons_fin[(cons_fin["filtro"]=="G") & (cons_fin["dest_clean"]=="S/TR") ].groupby(["HS6_d12" , "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_bk"}).reset_index(drop = True)
cons_fin_G_median_bk = cons_fin[(cons_fin["filtro"]=="G") & (cons_fin["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_bk"}).reset_index(drop = True)
cons_fin_G_mad_ci = cons_fin[(cons_fin["filtro"]=="G") & (cons_fin["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_ci"}).reset_index(drop = True)
cons_fin_G_median_ci = cons_fin[(cons_fin["filtro"]=="G") & (cons_fin["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_decl", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_ci"}).reset_index(drop = True)

cons_fin_G = pd.merge(cons_fin[(cons_fin["filtro"]=="G")  & (cons_fin["dest_clean"]=="CONS&Otros")] , cons_fin_G_mad_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left", left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
cons_fin_G = pd.merge(cons_fin_G , cons_fin_G_median_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
cons_fin_G = pd.merge(cons_fin_G , cons_fin_G_mad_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])
cons_fin_G = pd.merge(cons_fin_G , cons_fin_G_median_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_decl"], right_on = ["HS6_d12", "uni_decl"])

cons_fin_G ["mad_bk"] = np.where(cons_fin_G ["mad_bk"]==0, 0.001, cons_fin_G ["mad_bk"]  )
cons_fin_G ["mad_ci"] = np.where(cons_fin_G ["mad_ci"]==0, 0.001, cons_fin_G ["mad_ci"]  )
cons_fin_G ["z_score_bk"] = cons_fin_G .apply(lambda x: 0.6745*((x["metric"]-x["median_bk"]))/(x["mad_bk"]), axis =1 )
cons_fin_G ["z_score_ci"] = cons_fin_G .apply(lambda x: 0.6745*((x["metric"]-x["median_ci"]))/(x["mad_ci"]), axis =1 )

cons_fin_G["ue_dest"] = np.where(np.abs(cons_fin_G["z_score_ci"]) > np.abs(cons_fin_G["z_score_bk"]), "BK", "CI" ) 
# cons_fin_G["ue_dest"].value_counts() 

cons_fin_G_no_cons= cons_fin[(cons_fin["filtro"]=="G")  & (cons_fin["dest_clean"]!="CONS&Otros")]
cons_fin_G_no_cons["ue_dest"] = np.where(cons_fin_G_no_cons["dest_clean"]== "S/TR","BK", "CI" ) 
# cons_fin_G_no_cons["ue_dest"].value_counts()

cons_fin_G_clasif = pd.concat([cons_fin_G_no_cons, cons_fin_G.drop(["median_bk", "median_ci",  "mad_bk", "mad_ci", "z_score_bk", "z_score_ci"],axis =1 )], axis = 0) 
# cons_fin_G_clasif  ["ue_dest"].value_counts()

cons_fin_clasif =pd.concat([cons_fin_D, cons_fin_G_clasif, cons_fin[cons_fin["filtro"].str.contains("A|B|C|E")]  ])
cons_fin_clasif [["filtro", "ue_dest"]].value_counts()#.sum()


# =============================================================================
# Consistencia de diagramas
# =============================================================================
# len(impo_bec_ci) + len(impo_bec_bk) + len(impo_bec_cons) == len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )
len(impo_bec_ci) + len(data_not_clasif) + len(data_clasif_ue_dest) + len(impo_bec_cons) == len(join_impo_clae) - len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )


# impo_ue_dest = pd.concat([pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha", 'metric', 'ue_dest', 'mad', 'median', 'z_score'], axis = 1), bk], axis =0)
cicf_ue_dest = pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha",  'mad', 'median', 'z_score'], axis = 1) #, bk], axis =0)
cicf_ue_dest["precio_kilo"] =  cicf_ue_dest["valor"]/cicf_ue_dest["kilos"]

# bk_ue_dest = pd.read_csv("../data/resultados/bk_con_ue_dest.csv")
bk_ue_dest = data_clasif_ue_dest.copy().drop(['HS4', 'HS4Desc', 'HS6Desc', "BEC5Category"], 1)

# bk_sin_ue_dest = pd.read_csv("../data/resultados/bk_sin_ue_dest.csv")
bk_sin_ue_dest = data_not_clasif.drop(['HS4', 'HS4Desc', 'HS6Desc', "BEC5Category"], 1)


(len(join_impo_clae)-  len(impo_bec[impo_bec["BEC5EndUse"].isnull()] )) - ( len(bk_sin_ue_dest ) + len(bk_ue_dest)+ len(cicf_ue_dest) )

data_model = pd.concat([bk_sin_ue_dest , bk_ue_dest, cicf_ue_dest ], axis = 0) 
data_model ['HS6'] = data_model ['HS6'].astype("str")
# data_model.to_csv("../data/resultados/data_modelo_diaria.csv", index = False)
len(join_impo_clae) == (len(data_model) + len(impo_bec[impo_bec["BEC5EndUse"].isnull()] ))

# len(data_model)  - data_model["ue_dest"].value_counts().sum()


# =============================================================================
# Preprocesamiento de Datos para el modelo
# =============================================================================
# data_model= pd.read_csv("../data/resultados/data_modelo_diaria.csv")

data_model.info()
data_model["ue_dest"].value_counts()

data_model["actividades"] = data_model["letra1"]+data_model["letra2"]+data_model["letra3"]+data_model["letra4"]+data_model["letra5"]+data_model["letra6"]
data_model["act_ordenadas"] = data_model["actividades"].apply(lambda x: "".join(sorted(x ))) #"".join(sorted(data_model["actividades"]))

#preprocesamiento etiquetados
cols =  [ "HS6",  
         'valor',  'kilos', "precio_kilo" , 
         "letra1",	"letra2",	"letra3", 	"letra4", 	"letra5",	"letra6"	,
         "act_ordenadas",
         "uni_est",	"cant_est",	"uni_decl",	"cant_decl",  
         "metric" ,
         "ue_dest"]

data_model = data_model[cols]

#Filtros de columnas
cat_col = list(data_model.select_dtypes(include=['object']).columns)
cat_col.pop(-1)
num_col = list(data_model.select_dtypes(include=['float', "int64" ]).columns)

data_pre = pd.concat( [ str_a_num(data_model[cat_col]) , data_model[num_col], data_model["ue_dest"] ], axis = 1  )

# datos etiquetados
data_train = data_pre[data_pre ["ue_dest"] != "?" ]
data_train["bk_dummy"] = data_train["ue_dest"].map({"BK": 1, "CI": 0})
data_train.drop("ue_dest", axis = 1, inplace = True)

# datos no etiquetados
data_to_clasif = data_pre[data_pre["ue_dest"] == "?" ]
data_to_clasif.drop("ue_dest", axis = 1, inplace = True)

len(data_pre) == (len(data_train) + len(data_to_clasif))

# exportacion de datos
data_train.to_csv("../data/resultados/data_train_test.csv", index=False)
data_to_clasif .to_csv("../data/resultados/data_to_pred.csv", index=False)



