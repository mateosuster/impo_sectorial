# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:34:56 2021

@author: mateo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick




def impo_total(matriz_sisd, dic_propio, sectores_desc =False, largo_actividad=20):

    letras_ciiu= dic_propio[["propio_letra_2", "desc"]].drop_duplicates().rename(columns = {"propio_letra_2": "letra"}).dropna().sort_values("letra")
    letras_ciiu["desc"] = letras_ciiu["desc"].str.slice(0,largo_actividad)
    letras_ciiu= letras_ciiu[letras_ciiu["letra"]!= "CONS"]
    letras_ciiu = pd.concat([letras_ciiu , pd.DataFrame({"letra":"CONS", "desc": "Consumo"}, index= [None])], axis = 0)
    # if sectores_desc == True:
    #     indice = z.index
    #     letra =  indice.values
    # else:
    #     indice = letras_ciiu.desc.values
    #     letra = letras_ciiu.letra.values
    
    indice = letras_ciiu.desc.values
    letra = letras_ciiu.letra.values

    # matriz_sisd = matriz_sisd_ci
    # transformacion a array de np
    z_visual = matriz_sisd.to_numpy()
    
    #total y diagonal 
    col_sum  = np.nansum(z_visual , axis=0)
    diagonal = np.diag(z_visual)
   
    #importaciones totales (ordenadas)
    impo_tot_sec = pd.DataFrame({"impo_tot": col_sum, "letra":letra}, index=indice)  
    
    #diagonal sobre total col y comercio sobre total
    diag_total_col = diagonal/col_sum
    g_total_col = z_visual[29][:]/col_sum
    comercio_y_propio = pd.DataFrame({"Propio": diag_total_col*100 , 'Comercio': g_total_col*100} , index = indice )
        
    comercio_y_propio.iloc[29, 1] = 0
    
    return impo_tot_sec, comercio_y_propio.sort_values(by = 'Propio', ascending = False)


    
def graficos(dic_propio, impo_tot_sec, comercio_y_propio,  ue_dest, largo_actividad=20):
   
    titulo =np.nan 
   #parametro para graficos
    params = {'legend.fontsize': 20,
              'figure.figsize': (20, 10),
             'axes.labelsize': 15,
             'axes.titlesize': 30,
              'xtick.labelsize':15,
              'ytick.labelsize':20
             } 
    plt.rcParams.update(params)

    #diccionario para el eje x y ordenamiento de los datos
    # dic_graf = pd.concat([dic_propio[["propio_letra_2", "desc"]], pd.DataFrame({"propio_letra_2": ["CONS"], "desc": ["Consumo"]})], axis=0).drop_duplicates()
    dic_graf = dic_propio[["propio_letra_2", "desc"]].drop_duplicates()
    
    impo_tot_sec = pd.merge(impo_tot_sec,dic_graf , how = "left", left_on ="letra", right_on="propio_letra_2")#.drop("propio_letra_2",1)
    impo_tot_sec["desc"] =  impo_tot_sec["desc"].str.slice(0,largo_actividad)
    impo_tot_sec.set_index(["desc"],inplace=True)
    impo_tot_sec.sort_values("impo_tot",ascending=False, inplace=True)

    #titulo
    if ue_dest == "bk":
        titulo = "Bienes de Capital"
    elif ue_dest == "ci":
        titulo = "Consumos Intermedios"
    
    ##### grafico 1
    #posiciones para graf
    y_pos = np.arange(len(impo_tot_sec.index))
    
    values = impo_tot_sec.iloc[:,0]/(10**6)
    plt.bar(y_pos ,values  )
    plt.xticks(y_pos , impo_tot_sec.index, rotation = 90)
    plt.yticks(np.arange(0, values.max(), 1000))
    plt.title("Importaciones de "+ titulo + " destinadas a cada sector")#, fontsize = 30)
    plt.ylabel("Millones de USD")
    plt.xlabel("Sector \n \n Fuente: CEPXXI en base a Aduana, AFIP y UN Comtrade")
    # plt.subplots_adjust(bottom=0.7,top=0.83)
    plt.grid()
    plt.tight_layout()
    plt.savefig('../data/resultados/impo_totales_letra_'+ue_dest+'.png')
    plt.show()

    ##### grafico 2
    #graf division comercio y propio
    ax = comercio_y_propio.plot(kind = "bar", rot = 90,
                                stacked = True, ylabel = "", 
                                xlabel = "Sector \n \n Fuente: CEPXXII en base a Aduana, AFIP y UN Comtrade")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(pad=3)
    plt.grid()
    plt.title( "Sector abastecedor de importaciones de " + titulo + " (en porcentaje)")#,  fontsize = 30)
    plt.savefig('../data/resultados/comercio_y_propio_letra_'+ue_dest+'.png')
    plt.show()

def top_5(asign_pre_matriz, ncm12_desc, impo_tot_sec, dic_propio, bien, n=5):

    x = asign_pre_matriz.groupby(["hs6_d12", "sd"], as_index=False)['valor_pond'].sum("valor_pond")
    top_5_impo = x.reset_index(drop = True).sort_values(by = ["sd", "valor_pond"],ascending = False)
    top_5_impo["HS6"] = top_5_impo["hs6_d12"].str.slice(0,6).astype(int)
    top_5_impo  = top_5_impo.groupby(["sd"], as_index = True).head(n)
    top_5_impo  = pd.merge(left=top_5_impo, right=ncm12_desc, left_on="hs6_d12", right_on="HS_12d", how="left").drop("HS_12d", axis=1)
    top_5_impo  = pd.merge(top_5_impo  , impo_tot_sec, left_on="sd", right_on="letra", how = "left")
    top_5_impo["impo_relativa"] = top_5_impo["valor_pond"]/top_5_impo["impo_tot"] 
    top_5_impo["short_name"] = top_5_impo["hs6_d12_desc"].str.slice(0,15)
    
    top_5_impo = pd.merge(top_5_impo, dic_propio[["propio_letra_2", "desc"]].drop_duplicates(), how = "left", left_on="sd", right_on = "propio_letra_2").drop("propio_letra_2", 1)
    
    top_5_impo.to_excel("../data/resultados/top"+str(n)+"_impo_"+bien+".xlsx")
    top_5_impo.to_csv("../data/resultados/top"+str(n)+"_impo_"+bien+".csv", index = False)
    
    return top_5_impo


def def_top_hs(asign_pre_matriz, ncm12_desc, bien):
    top_productos = asign_pre_matriz.groupby(["hs6_d12"], as_index=False)['valor_pond'].sum("valor_pond").sort_values("valor_pond", ascending=False).iloc[0:50]
    top_productos  = pd.merge(left=top_productos , right=ncm12_desc, left_on="hs6_d12", right_on="HS_12d", how="left").drop("HS_12d", axis=1)
    top_productos.to_csv("../data/resultados/top_productos_"+bien+".csv", index=False)
    return top_productos  

def def_top_sd_de_top_hs(asign_pre_matriz, ncm12_desc, dic_propio,top_productos, bien):
    prpal_sd = asign_pre_matriz[asign_pre_matriz["hs6_d12"].isin(top_productos["hs6_d12"])].groupby(["sd", "hs6_d12"], as_index=False)["valor_pond"].sum().sort_values(["valor_pond"], ascending = False)
    prpal_sd  = prpal_sd .groupby(["hs6_d12"], as_index = False).head(5).sort_values(["hs6_d12", "valor_pond"], ascending=False)
    prpal_sd  = pd.merge(left=prpal_sd  , right=ncm12_desc, left_on="hs6_d12", right_on="HS_12d", how="left").drop("HS_12d", axis=1)
    prpal_sd   = pd.merge(prpal_sd  , dic_propio[["propio_letra_2", "desc"]].drop_duplicates(), how = "left", left_on="sd", right_on = "propio_letra_2").drop("propio_letra_2", 1)
    prpal_sd.to_csv("../data/resultados/principales_destinos_del_top_hs_"+bien+".csv", index = False)
    return prpal_sd

def def_top_cuits(asign_pre_matriz, dic_propio, bien, cuits_desc ):
    top_cuits= asign_pre_matriz.groupby(["cuit", "sd"], as_index=False)['valor_pond'].sum("valor_pond").sort_values("sd", ascending=False)#.iloc[0:50]
    top_cuits =  top_cuits.groupby(["sd"], as_index=False).head(10)
    top_cuits   = pd.merge(top_cuits  ,cuits_desc , how = "left", left_on="cuit", right_on = "cuit")
    top_cuits   = pd.merge(top_cuits  , dic_propio[["propio_letra_2", "desc"]].drop_duplicates(), how = "left", left_on="sd", right_on = "propio_letra_2").drop("propio_letra_2", 1)    
    top_cuits.to_csv("../data/resultados/principales_cuits_"+bien+".csv", index =False)
    return top_cuits 

def def_top_cuit_de_top_hs(asign_pre_matriz, ncm12_desc, dic_propio, top_productos, bien):
    prpal_sd = asign_pre_matriz[asign_pre_matriz["hs6_d12"].isin(top_productos["hs6_d12"])].groupby(["cuit", "hs6_d12"], as_index=False)["valor_pond"].sum().sort_values(["cuit","hs6_d12", "valor_pond"], ascending = False)
    prpal_sd  = prpal_sd .groupby(["hs6_d12"], as_index = False).head(10).sort_values(["hs6_d12", "valor_pond"], ascending=False)
    prpal_sd  = pd.merge(left=prpal_sd  , right=ncm12_desc, left_on="hs6_d12", right_on="HS_12d", how="left").drop("HS_12d", axis=1)
    # prpal_sd   = pd.merge(prpal_sd  , dic_propio[["propio_letra_2", "desc"]].drop_duplicates(), how = "left", left_on="sd", right_on = "propio_letra_2").drop("propio_letra_2", 1)
    prpal_sd.to_csv("../data/resultados/principales_cuits_top_hs_"+bien+".csv", index = False)  
    return prpal_sd















############################################################################################################
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