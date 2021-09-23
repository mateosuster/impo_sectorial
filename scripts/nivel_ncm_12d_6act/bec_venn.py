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

impo_d12["dest_clean"] = impo_d12["destinacion"].apply(lambda x: destinacion_limpio(x))
impo_d12["dest_clean"].value_counts()

impo_bec = pd.merge(impo_d12, bec[["HS6", "BEC5EndUse" ]], how= "left" , left_on = "HS6", right_on= "HS6" )
impo_bec[impo_bec["BEC5EndUse"].isnull()]


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

dest_a["filtro"] = "A" 
dest_b["filtro"] = "B" 
dest_c["filtro"] = "C" 
dest_d["filtro"] = "D" 
dest_e["filtro"] = "E"                                          
dest_f["filtro"] = "F"
dest_g["filtro"] = "G"  

bk = pd.concat( [dest_a, dest_b, dest_c, dest_d,  dest_e,  dest_f, dest_g], axis = 0)


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
cons_int_D_mad= cons_int[(cons_int["filtro"]=="D") & (cons_int["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
cons_int_D_median = cons_int[(cons_int["filtro"]=="D") & (cons_int["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
cons_int_D = pd.merge(cons_int[(cons_int["filtro"]=="D") & (cons_int["dest_clean"]=="CONS&Otros") ] , cons_int_D_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_est", "filtro"], right_on = ["HS6_d12", "uni_est", "filtro"])
cons_int_D = pd.merge(cons_int_D, cons_int_D_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_est", "filtro"], right_on = ["HS6_d12", "uni_est", "filtro"])

cons_int_D["mad"] = np.where(cons_int_D["mad"]==0, 0.001, cons_int_D["mad"]  )
cons_int_D["z_score"] = cons_int_D.apply(lambda x: 0.6745*((x["metric"]-x["median"]))/(x["mad"]), axis =1 )
cons_int_D["ue_dest"] = np.where(cons_int_D["z_score"] > 3.5, "BK", "CI" ) 

cons_int_D_st = cons_int[(cons_int["filtro"]=="D") & (cons_int["dest_clean"]=="S/TR") ]
cons_int_D_st["ue_dest"] = "BK"

cons_int_D = pd.concat([cons_int_D, cons_int_D_st], axis = 0)
# cons_int_D["ue_dest"].value_counts()

# CASO G
# cons_int[cons_int["filtro"]=="G"].value_counts("dest_clean")
cons_int_G_mad_bk = cons_int[(cons_int["filtro"]=="G") & (cons_int["dest_clean"]=="S/TR") ].groupby(["HS6_d12" , "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_bk"}).reset_index(drop = True)
cons_int_G_median_bk = cons_int[(cons_int["filtro"]=="G") & (cons_int["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_bk"}).reset_index(drop = True)
cons_int_G_mad_ci = cons_int[(cons_int["filtro"]=="G") & (cons_int["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_ci"}).reset_index(drop = True)
cons_int_G_median_ci = cons_int[(cons_int["filtro"]=="G") & (cons_int["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_ci"}).reset_index(drop = True)

cons_int_G = pd.merge(cons_int[(cons_int["filtro"]=="G")  & (cons_int["dest_clean"]=="CONS&Otros")] , cons_int_G_mad_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left", left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
cons_int_G = pd.merge(cons_int_G , cons_int_G_median_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
cons_int_G = pd.merge(cons_int_G , cons_int_G_mad_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
cons_int_G = pd.merge(cons_int_G , cons_int_G_median_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])

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
cons_fin_D_mad= cons_fin[(cons_fin["filtro"]=="D") & (cons_fin["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad"}).reset_index(drop = True)
cons_fin_D_median = cons_fin[(cons_fin["filtro"]=="D") & (cons_fin["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median"}).reset_index(drop = True)
cons_fin_D = pd.merge(cons_fin[(cons_fin["filtro"]=="D") & (cons_fin["dest_clean"]=="CONS&Otros") ] , cons_fin_D_mad.drop("dest_clean", axis =1), how = "left", left_on = ["HS6_d12", "uni_est", "filtro"], right_on = ["HS6_d12", "uni_est", "filtro"])
cons_fin_D = pd.merge(cons_fin_D, cons_fin_D_median.drop("dest_clean", axis =1), how = "left" ,left_on = ["HS6_d12", "uni_est", "filtro"], right_on = ["HS6_d12", "uni_est", "filtro"])

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
cons_fin_G_mad_bk = cons_fin[(cons_fin["filtro"]=="G") & (cons_fin["dest_clean"]=="S/TR") ].groupby(["HS6_d12" , "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_bk"}).reset_index(drop = True)
cons_fin_G_median_bk = cons_fin[(cons_fin["filtro"]=="G") & (cons_fin["dest_clean"]=="S/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_bk"}).reset_index(drop = True)
cons_fin_G_mad_ci = cons_fin[(cons_fin["filtro"]=="G") & (cons_fin["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : median_abs_deviation(x)).rename(columns = {"metric": "mad_ci"}).reset_index(drop = True)
cons_fin_G_median_ci = cons_fin[(cons_fin["filtro"]=="G") & (cons_fin["dest_clean"]=="C/TR") ].groupby(["HS6_d12", "dest_clean", "uni_est", "filtro"], as_index= False)["metric"].apply(lambda x : x.median()).rename(columns = {"metric": "median_ci"}).reset_index(drop = True)

cons_fin_G = pd.merge(cons_fin[(cons_fin["filtro"]=="G")  & (cons_fin["dest_clean"]=="CONS&Otros")] , cons_fin_G_mad_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left", left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
cons_fin_G = pd.merge(cons_fin_G , cons_fin_G_median_bk.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
cons_fin_G = pd.merge(cons_fin_G , cons_fin_G_mad_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])
cons_fin_G = pd.merge(cons_fin_G , cons_fin_G_median_ci.drop(["dest_clean", "filtro"], axis =1) , how = "left" ,left_on = ["HS6_d12", "uni_est"], right_on = ["HS6_d12", "uni_est"])

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
# cons_fin_clasif ["ue_dest"].value_counts()#.sum()


# =============================================================================
# Consistencia de diagramas
# =============================================================================
len(impo_bec_ci) + len(impo_bec_bk) + len(impo_bec_cons)

impo_ue_dest = pd.concat([pd.concat([cons_fin_clasif, cons_int_clasif], axis = 0).drop(["brecha", 'metric', 'ue_dest', 'mad', 'median', 'z_score'], axis = 1), bk], axis =0)
