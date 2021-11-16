# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:34:56 2021

@author: mateo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def sectores():
    sectores_desc = {"A":	"Agricultura",  "B":	"Minas y canteras", "C":	"Industria", "D": "Energía",
                     "E":	"Agua y residuos", "F":	"Construcción", "G": "Comercio", "H":	"Transporte",
                     "I":	"Alojamiento", "J":	"Comunicaciones", "K":	"Serv. financieros","L":	"Serv. inmobiliarios",
                     "M":	"Serv. profesionales", "N":	"Serv. apoyo", "O":	"Sector público", "P":	"Enseñanza",
                     "Q":	"Serv. sociales", "R":	"Serv. culturales", "S":	"Serv. personales", "T":	"Serv. doméstico",
                     "U": "Serv. organizaciones" , "CONS": "Consumo" }
    return sectores_desc


def impo_total(z, sectores_desc =False):
    indice = z.index
    # transformacion a array de np
    z_visual = z.to_numpy()
    
    #diagonal y totales col y row
    col_sum  = np.nansum(z_visual , axis=0)
    
    # sectores
    # sectores_desc.pop("U")
    # sectores = pd.DataFrame( { "desc":sectores_desc.values(), "letra": sectores_desc.keys() })

    #importaciones totales (ordenadas)
    # impo_tot_sec = pd.DataFrame({"impo_tot": col_sum, "letra":sectores_desc.keys() }, index=sectores_desc.values())  
    impo_tot_sec = pd.DataFrame({"impo_tot": col_sum, "letra":indice.values}, index=indice)  
    # impo_tot_sec = pd.DataFrame({"impo_tot": col_sum } )  
    # impo_tot_sec.sort_values(by = "impo_tot", ascending= False, inplace = True)
    
    return impo_tot_sec

def impo_comercio_y_propio(z, sectores_desc = False):
    indice = z.index
    
    # transformacion a array de np
    z_visual = z.to_numpy()
    
    #diagonal y totales col y row
    diagonal = np.diag(z_visual)
    col_sum  = np.nansum(z_visual , axis=0)
    
    #diagonal sobre total col y comercio sobre total
    diag_total_col = diagonal/col_sum
    g_total_col = z_visual[29][:]/col_sum
    comercio_y_propio = pd.DataFrame({"Propio": diag_total_col*100 , 'Comercio': g_total_col*100} , index = indice )
    return comercio_y_propio.sort_values(by = 'Propio', ascending = False) 
    
def graficos(matriz_sisd, impo_tot_sec, comercio_y_propio):
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
    # y_pos = np.arange(len(sectores_desc.values())) 
    y_pos = np.arange(len(matriz_sisd.index.values)) 
    
    plt.bar(y_pos , impo_tot_sec.iloc[:,0]/(10**6) )
    plt.xticks(y_pos , impo_tot_sec.index, rotation = 45)
    plt.title("Importaciones de bienes de capital destinadas a cada sector")#, fontsize = 30)
    plt.ylabel("Millones de USD")
    plt.xlabel("Sector \n \n Fuente: CEPXXI en base a Aduana, AFIP y UN Comtrade")
    # plt.subplots_adjust(bottom=0.7,top=0.83)
    plt.tight_layout()
    plt.show()
    plt.savefig('../data/resultados/impo_totales_letra.png')
    

    ##### grafico 2
    #graf division comercio y propio
    ax = comercio_y_propio.plot(kind = "bar", rot = 75,
                                stacked = True, ylabel = "", 
                                xlabel = "Sector \n \n Fuente: CEPXXII en base a Aduana, AFIP y UN Comtrade")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(pad=3)
    plt.title( "Sector abastecedor de importaciones de bienes de capital (en porcentaje)")#,  fontsize = 30)
    
    plt.savefig('../data/resultados/comercio_y_propio_letra.png')


def top_5(asign_pre_matriz, letras_ciiu , ncm12_desc, impo_tot_sec):

    x = asign_pre_matriz.groupby(["hs6_d12", "sd"], as_index=False)['valor_pond'].sum("valor_pond")
    top_5_impo = x.reset_index(drop = True).sort_values(by = ["sd", "valor_pond"],ascending = False)
    top_5_impo["HS6"] = top_5_impo["hs6_d12"].str.slice(0,6).astype(int)
    top_5_impo  = top_5_impo.groupby(["sd"], as_index = True).head(5)
    # top_5_impo  = pd.merge(left=top_5_impo, right=letras_ciiu, left_on="sd", right_on="letra", how="left").drop(["sd"], axis=1) # ACA OBTENIA LA DESCLASIFICACION
    # top_5_impo  = pd.merge(left=top_5_impo, right=bec[["HS6","HS6Desc"]], left_on="HS6", right_on="HS6", how="left").drop("HS6", axis=1)
    top_5_impo  = pd.merge(left=top_5_impo, right=ncm12_desc, left_on="hs6_d12", right_on="HS_12d", how="left").drop("HS_12d", axis=1)
    top_5_impo  = pd.merge(top_5_impo  , impo_tot_sec, left_on="sd", right_on="letra", how = "left")
    top_5_impo["impo_relativa"] = top_5_impo["valor_pond"]/top_5_impo["impo_tot"] 
    top_5_impo["short_name"] = top_5_impo["hs6_d12_desc"].str.slice(0,15)

    return top_5_impo


def predo_mca(matriz_sisd_final, do):
    #groupby para el mca producto-. Poner en el archivo de visualización
    matriz_mca = matriz_sisd_final.copy()
    matriz_mca = matriz_mca.drop(["cuit", "si", "ue_dest"], axis=1)
    matriz_mca = pd.pivot_table(matriz_sisd_final, values='valor_pond', index=['hs6'], columns=['sd'], 
                                aggfunc= do, fill_value=0)
    
    # matriz_mca.groupby(["hs6", "sd"]).count().reset_index() 
    
    
    # matriz_mca.set_index("hs6", inplace=True)
    matriz_mca  = matriz_mca.transpose()
    
    return matriz_mca


def predo_bce_cambiario(bce_cambiario):

    bce_cambiario.drop(bce_cambiario.columns[16:], axis=1, inplace=True)
    bce_cambiario.rename(columns= {"Años": "anio", "ANEXO": "partida", "Denominación": "sector" }, inplace= True)
    
    #conversion a numeros
    to_num_cols= bce_cambiario.columns[4:]
    bce_cambiario[to_num_cols] = bce_cambiario[to_num_cols].apply(pd.to_numeric,errors='coerce')
    
    #filtro
    bce_cambiario_filter = bce_cambiario[(bce_cambiario["anio"] ==2017) & (bce_cambiario["partida"].isin([7,8,10]) )].drop([ "anio", "partida", "C-V"], axis=1) 
    impo_tot_bcra = bce_cambiario_filter.groupby( "sector", as_index = True).sum().sum(axis= 1).reset_index().rename(columns= { 0: "impo_bcra"} )
    
    #suma de importaciones
    #quedaron afuera informaticam oleaginosas y cerealeros
    A= impo_tot_bcra.iloc[[0]].sum() #agro
    B= impo_tot_bcra.iloc[[22]].sum() #minas
    C = impo_tot_bcra.iloc[[2,11,12,13,14,20,23]].sum() #industria
    D= impo_tot_bcra.iloc[[6,9]].sum() #energia 
    E= impo_tot_bcra.iloc[[1]].sum()  #agua
    F= impo_tot_bcra.iloc[[5]].sum()  #construccion
    G= impo_tot_bcra.iloc[[3]].sum() #comercio
    H= impo_tot_bcra.iloc[[27]].sum() #transporte
    I= impo_tot_bcra.iloc[[8,28]].sum() #hoteles y gastronomia
    J= impo_tot_bcra.iloc[[4]].sum() #comunicaciones
    K= impo_tot_bcra.iloc[[7,25]].sum() #finanzas
    O= impo_tot_bcra.iloc[[24]].sum() #sector publico
    R= impo_tot_bcra.iloc[[8]].sum() #serv culturales
    
    impo_bcra_letra =  pd.DataFrame([A,B,C,D,E,F,G,H,I,J,K,O,R], index= ["A","B","C","D","E","F","G","H","I","J","K","O","R"]).drop("sector", axis = 1).reset_index().rename(columns= {"index": "letra", 0: "impo_bcra"} )
    
    return impo_bcra_letra


def impo_ciiu_letra(isic, dic_ciiu,join_impo_clae_bec_bk ):

    isic=isic.iloc[:,[0,2]]
    isic.columns= ["HS6", "ISIC"]
    
    dic_ciiu = dic_ciiu.iloc[:, [0,-2,-1] ]
    dic_ciiu.columns = ["ISIC", "letra", "letra_desc"]
    
    ciiu = pd.merge(isic, dic_ciiu, how= "left" , left_on ="ISIC", right_on ="ISIC")
    
    impo_ciiu =pd.merge(join_impo_clae_bec_bk[["HS6", "valor"]], ciiu, how = "left" ,right_on="HS6", left_on="HS6")
    impo_ciiu_letra =impo_ciiu[impo_ciiu["letra"].notnull()].groupby("letra")["valor"].sum()
    
    as_list = impo_ciiu_letra.index.tolist()
    idx = as_list.index("D")
    as_list[idx] = "C"
    impo_ciiu_letra.index = as_list
    
    impo_ciiu_letra = pd.DataFrame(impo_ciiu_letra).reset_index() 
    impo_ciiu_letra.columns = ["letra", "impo_ciiu"]
    
    return impo_ciiu_letra