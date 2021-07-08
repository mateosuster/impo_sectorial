# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:04:54 2021

@author: igalk
"""
import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Bases import *
from procesamiento import *
from matriz import *


# =============================================================================
# Directorio de trabajo
# =============================================================================

#Mateo
os.chdir("C:/Archivos/repos/impo_sectorial/scripts")
os.getcwd()

#############################################
# Cargar bases con las que vamos a trabajar #
#############################################
impo_17 = pd.read_csv(  "data/IMPO_2017.csv", sep=";")
clae = pd.read_csv( "data/clae_nombre.csv")
comercio = pd.read_csv("data/comercio_clae.csv", encoding="latin1")
cuit_clae = pd.read_csv( "data/cuit 2017 impo_con_actividad.csv")
bec = pd.read_csv( "data/HS2012-17-BEC5 -- 08 Nov 2018.csv")
bec_to_clae = pd.read_csv("data/bec_to_clae.csv")

# parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")

industria_2d = clae[clae["letra"]=="C"][["clae2", "clae2_desc"]].drop_duplicates() 
industria_2d["clae2"] = industria_2d["clae2"].astype(str).str.slice(0,2)

desc_2d = ["Alimentos", "Bebidas", "Tabaco", "Textiles", "Prendas de vestir", "Productos de cuero",
           "Productos de madera", "Productos de papel", "Imprentas", "Productos de coque",
           "Químicos", "Farmacéuticos", "Caucho y vidrio", "Minerales no metálicos", "Hierro y acero",
           "Productos metálicos", "Electrónicos", "Maquinaria", "Aparatos uso doméstico", 
           "Equipos de automotores", "Equipo de transporte", "Muebles", "Manufacturas diversas", "Reparación maquinaria"]
industria_2d.insert(2, "desc", desc_2d)

comercio_2d = clae[clae["letra"]=="G"][["clae2", "clae2_desc"]].drop_duplicates() 
comercio_2d["clae2"] = comercio_2d["clae2"].astype(str).str.slice(0,2)

#############################################
#           preparación bases               #
#############################################

predo_impo_17(impo_17)
letras = predo_sectores_nombres(clae)
comercio = predo_comercio(comercio, clae)

cuit_clae_6d = predo_cuit_clae_6d(cuit_clae)

cuit_personas = predo_cuit_clae(cuit_clae_6d, "personas")
cuit_empresas = predo_cuit_clae(cuit_clae_6d, "empresas")
bec_bk = predo_bec_bk(bec, bec_to_clae)



#############################################
#                joins                      #
#############################################

join_impo_clae = def_join_impo_clae(impo_17, cuit_empresas)
join_impo_clae_bec_bk = def_join_impo_clae_bec(join_impo_clae, bec_bk)
join_impo_clae_bec_bk_comercio = def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio)


#############################################
#           Tabla de contingencia           #
#              producto-sector              #
#############################################

tabla_contingencia = def_contingencia(join_impo_clae_bec_bk_comercio)

#############################################
#      ponderación por ncm y letra          #
#############################################

join_impo_clae_bec_bk_comercio_pond = def_join_impo_clae_bec_bk_comercio_pond(join_impo_clae_bec_bk_comercio)

# join_final = def_calc_pond(join_impo_clae_bec_bk_comercio_pond,tabla_contingencia)
join_final = pd.read_csv("data/resultados/impo_con_ponderaciones_2d.csv")


#############################################
#         ASIGNACIÓN y MATRIZ               #
#############################################

#creamos df para guardar los insumos de la matriz
insumo_matriz = pd.DataFrame()
insumo_matriz ["cuit"]=""
insumo_matriz ["hs6"]=""
insumo_matriz ["valor_pond"]=""    
insumo_matriz ["si"]=""    
insumo_matriz ["sd"]=""
insumo_matriz ["ue_dest"]=""

# matriz_sisd = def_insumo_matriz(insumo_matriz, join_final)
# matriz_sisd.to_csv("data/matriz_pesada_2d.csv")
matriz_sisd = pd.read_csv("data/resultados/matriz_pesada_2d.csv")

#asignación por probabilidad de G-bk (insumo para la matriz)
matriz_sisd_final = def_matriz_c_prob(matriz_sisd)
matriz_sisd_final["si"] = matriz_sisd_final["si"].astype(str)


#matriz rotada
z = pd.pivot_table(matriz_sisd_final, values='valor_pond', index=['si'], columns=['sd'], aggfunc=np.sum, fill_value=0)

#filtro industria
z_industria = z.loc[:, industria_2d["clae2"]]


#busco las filas que faltan
# set(list(z.columns)).symmetric_difference( set(list( z.index ) ) )
# [x for x in list(z.index)  if x not in  list(z.columns) ] # OK pero son tipos de datos distintos


#############################################
#             Visualización                 #
#############################################

# transformacion a array de np
indu_np  = z_industria.to_numpy()

# totales col 
col_sum  = np.nansum(indu_np   , axis=0)

#importaciones totales (ordenadas)
impo_tot_sec = pd.DataFrame({"Importaciones totales": col_sum/(10**6), "clae_2d": list(industria_2d["clae2"])}, index=industria_2d["desc"])  
impo_tot_sec.sort_values(by = "Importaciones totales", ascending= False, inplace = True)

#parametro para graficos
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize': 30,
         'xtick.labelsize':12,
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
 

##### grafico 1
#posiciones para graf
y_pos = np.arange(len(industria_2d["desc"]))

plt.bar(y_pos , impo_tot_sec.iloc[:,0])
plt.xticks(y_pos , impo_tot_sec.index, rotation = 75)
plt.title("Importaciones de bienes de capital destinadas a industria manufacturera", fontsize = 30)
plt.ylabel("Millones de USD")
plt.xlabel("Sector \n \n Fuente: CEPXXI en base a Aduana y AFIP")
plt.tight_layout()
plt.savefig('data/resultados/impo_totales_2d.png')
    

##### grafico 2
#graf division comercio y propio

#diagonal sobre total col y comercio sobre total
z_industria_p_diag = z.loc[industria_2d["clae2"], industria_2d["clae2"]]
diagonal= np.diag(z_industria_p_diag )
diag_total_col = diagonal/col_sum

z_industria_g = z.loc[comercio_2d["clae2"], industria_2d["clae2"]]
sum_col_c_g = np.nansum(z_industria_g, axis = 0 )
g_total_col = sum_col_c_g /col_sum

import matplotlib.ticker as mtick

comercio_y_propio = pd.DataFrame({"Propio": diag_total_col*100 , 'Comercio': g_total_col*100} , index = industria_2d["desc"])


ax = comercio_y_propio.sort_values(by = 'Propio', ascending = False).plot(kind = "bar",  rot = 75,
                                                                     stacked = True, ylabel = "%", xlabel = "Sector \n \n Fuente: CEPXXI en base a Aduana y AFIP")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
plt.title("Sector abastecedor de importaciones de bienes de capital (en porcentaje)", fontsize = 30)
plt.tight_layout(pad=3)
plt.savefig('data/resultados/comercio_y_propio_2d.png')


##### grafico 3
# Top 5 de importaciones de cada sector


x = matriz_sisd_final[matriz_sisd_final["sd"].isin(industria_2d["clae2"])].groupby(["hs6", "sd"], as_index=False)['valor_pond'].sum("valor_pond")
top_5_impo = x.reset_index(drop = True).sort_values(by = ["sd", "valor_pond"],ascending = False)
top_5_impo  = top_5_impo.groupby(["sd"], as_index = True).head(5)
top_5_impo  = pd.merge(left=top_5_impo, right=industria_2d, left_on="sd", right_on="clae2", how="left").drop(["sd", "desc"], axis=1)
top_5_impo  = pd.merge(left=top_5_impo, right=bec[["HS6","HS6Desc"]], left_on="hs6", right_on="HS6", how="left").drop("HS6", axis=1)
top_5_impo  = pd.merge(top_5_impo  , impo_tot_sec, left_on="clae2", right_on="clae_2d", how = "left")
top_5_impo["impo_relativa"] = top_5_impo["valor_pond"]/(top_5_impo["Importaciones totales"] * 10**6) 

top_5_impo.to_excel("data/resultados/top_5_impo_2d.xlsx")


#### grafico 4
impo_tot_sec.drop("clae_2d", axis=1, inplace= True)
x1 = impo_tot_sec.loc[["Alimentos", "Bebidas", "Tabaco"]].sum().reset_index(drop=True)
x2 = impo_tot_sec.loc[["Equipos de automotores", "Equipo de transporte"]].sum().reset_index(drop=True)
x3 = impo_tot_sec.loc[["Químicos", "Productos de coque", "Caucho y vidrio"]].sum().reset_index(drop=True)
x4 = impo_tot_sec.loc[["Textiles", "Prendas de vestir", "Productos de cuero"]].sum().reset_index(drop=True)
x5 = impo_tot_sec.loc[["Productos de papel", "Imprentas"]].sum().reset_index(drop=True)
x6 = impo_tot_sec.loc[["Maquinaria"]].sum().reset_index(drop=True)
x7 = impo_tot_sec.loc[["Hierro y acero", "Productos metálicos"]].sum().reset_index(drop=True)
x8 = impo_tot_sec.loc[["Minerales no metálicos"]].sum().reset_index(drop=True)


dic = {"sector": ["Alimentos, Bebidas y Tabaco", "Industria Automotriz",
                  "Industria Química, Caucho y Plástico", 
                  "Industria Textil y Curtidos", 
                  "Industria de Papel, Ediciones e Impresiones" ,
                  "Maquinarias y Equipos", 
                  "Metales Comunes y Elaboración",
                  "Productos Minerales no Metálicos (Cementos, Cerámicos y Otros)"] , 
       "impo_sisd": pd.concat([x1, x2, x3,x4,x5,x6,x7, x8])}


comparacion_bcra = pd.merge(pd.DataFrame(dic), impo_tot_bcra, 
                            left_on = "sector", right_on = "sector", how = "left")

comparacion_bcra["sector"] =comparacion_bcra["sector"].str.replace( "\(Cementos, Cerámicos y Otros\)", "")

comparacion_bcra.sort_values(by = 'impo_sisd', ascending = False).plot(x="sector", y = ["impo_sisd", "impo_bcra"], kind="bar", rot=15,
                 ylabel = "Millones de dólares", xlabel = "Sector \n \n Fuente: CEPXXI en base a BCRA, Aduana, AFIP y UN Comtrade. La estimación del BCRA no es exclusiva de Bienes de Capital")#,)
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(pad=3)
plt.title( "Importacion sectorial. Comparación de estimaciones",  fontsize = 30)
plt.legend(["SI-SD (BK)", "BCRA (BK+CI+CONS)"])
plt.savefig("data/resultados/comparacion_bcra_2d.png")


######## carga para CIIU
isic = pd.read_csv("data/JobID-64_Concordance_HS_to_I3.csv", encoding = "latin" )
isic=isic.iloc[:,[0,2]]
isic.columns= ["HS6", "ISIC"]

dic_ciiu = pd.read_excel("data/Diccionario CIIU3.xlsx")
dic_ciiu = dic_ciiu.iloc[:, [0 , 4,5,6] ]
dic_ciiu.columns = ["ISIC",  "ciiu_2d","ciiu_2desc",  "letra"]


ciiu = pd.merge(isic, dic_ciiu, how= "left" , left_on ="ISIC", right_on ="ISIC")

impo_ciiu =pd.merge(join_impo_clae_bec_bk[["HS6", "valor"]], ciiu, how = "left" ,right_on="HS6", left_on="HS6")
impo_ciiu["ciiu_2d"] = impo_ciiu["ciiu_2d"].astype(str).str.slice(0,2)

impo_ciiu_indu = impo_ciiu[impo_ciiu["ciiu_2d"].isin(industria_2d["clae2"])]

impo_ciiu_letra =impo_ciiu_indu.groupby("ciiu_2desc")["valor"].sum()

as_list = impo_ciiu_letra.index.tolist()
idx = as_list.index("D")
as_list[idx] = "C"
impo_ciiu_letra.index = as_list

impo_ciiu_letra = pd.DataFrame(impo_ciiu_letra).reset_index() 
impo_ciiu_letra.columns = ["letra", "impo_ciiu"]
