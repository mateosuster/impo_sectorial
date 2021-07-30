# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:44:28 2021

@author: igalk
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
import os 
os.getcwd()


# =============================================================================
# preprocesamiento
# =============================================================================


# def carga_de_bases(x):
#     impo_17 = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/IMPO_2017.csv", sep=";")
#     clae = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/clae_nombre.csv")
#     print ("hola")
#     return impo_17
# #     comercio = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/comercio_clae.xlsx")
# #     cuit_clae = pd.read_csv("C:/Users/igalk/OneDrive/Documentos/CEP/procesamiento impo/cuit 2017 impo_con_actividad.csv")
# #     bec = pd.read_excel( "/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/HS2012-17-BEC5 -- 08 Nov 2018.xlsx", sheet_name= "HS17BEC5" )
# #     #parts_acces  =pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/nomenclador_28052021.xlsx", names=None  , header=None )
# #     #transporte_reclasif  = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/resultados/bec_transporte (reclasificado).xlsx")
# #     bec_to_clae = pd.read_excel("C:/Archivos/Investigación y docencia/Ministerio de Desarrollo Productivo/balanza comercial sectorial/tablas de correspondencias/bec_to_clae.xlsx")
# #     return impo_17


def predo_impo_17(impo_17):
    impo_17['FOB'] = impo_17['FOB'].str.replace(",", ".")
    impo_17['FOB'] = impo_17['FOB'].astype(float)
    impo_17['CIF'] = impo_17['CIF'].str.replace(",", ".")
    impo_17['CIF'] = impo_17['CIF'].astype(float)
    impo_17.drop("FOB", axis=1, inplace=True)
    return impo_17.rename(columns = {"POSIC_SIM" : "HS6", 'CIF': "valor"} , inplace = True)


def predo_sectores_nombres(clae):
    letras_np = pd.unique(clae['letra'])
    letras_np= np.delete(letras_np, 0)
    letras_np= letras_np[~pd.isnull(letras_np)]
    letras = pd.merge(pd.DataFrame({'letra': letras_np}), clae[['letra', 'letra_desc']],
                      how = "left", left_on = "letra", right_on = "letra")
    letras.drop_duplicates(subset = 'letra', inplace = True)
    letras = pd.DataFrame(letras)
    
    cons = pd.DataFrame([{"letra": "CONS", "letra_desc": "CONSUMO"}] )
    letras  = pd.concat([letras, cons] , axis =0)
    
    return letras


def predo_comercio(comercio, clae):
    comercio.rename(columns = { 'Unnamed: 2': "clae6" , "G": "clae3", 
                           'COMERCIO   AL   POR   MAYOR   Y   AL   POR   MENOR;   REPARACIÓN   DE   VEHÍCULOS AUTOMOTORES Y MOTOCICLETAS' : "clae6_desc",
                            'Venta de vehículos': "vta_vehiculos",
                            'Vende BK': "vta_bk",
                            'Vende mayormente BK a otros sectores productivos o CF?': "vta_sec"}  , inplace = True )
    comercio["clae3"].fillna(comercio['Unnamed: 1'], inplace = True)
    comercio["clae3"].fillna( method = "ffill", inplace = True)
    del comercio['Unnamed: 1']
    comercio.drop(comercio.iloc[:,6: ], axis= 1, inplace= True) 
    comercio_reclasificado =  pd.merge( left =clae[clae["letra"] =="G"][["letra", "clae6", "clae6_desc"]], 
                                       right = comercio.drop(["clae6_desc", "clae3"], axis = 1), left_on="clae6", right_on = "clae6", how="left" ) 
    return comercio_reclasificado

def predo_cuit_clae(cuit_clae, tipo_cuit):
    cuit_personas = cuit_clae[(cuit_clae["letra1"].isnull()) & (cuit_clae["letra2"].isnull()) & (cuit_clae["letra3"].isnull())]
    cuit_empresas = cuit_clae[~cuit_clae['cuit'].isin(cuit_personas['cuit'])]
    
    if tipo_cuit == "personas":
        return cuit_personas 
        
    else:
        #reemplzao faltantes de letra1 con letra2, y faltantes letra 2 con letra 3
        cuit_empresas['letra1'].fillna(cuit_empresas['letra2'], inplace = True) 
        cuit_empresas['letra2'].fillna(cuit_empresas['letra3'], inplace = True) #innecesario, porque no cambia la cuenta (quien tiene NaN en letra 2 tambien tiene NaN letra 3)
        
        #completo relleno de faltantes con letra 1
        cuit_empresas['letra2'].isnull().sum()
        cuit_empresas['letra3'].isnull().sum()
        cuit_empresas['letra2'].fillna(cuit_empresas['letra1'], inplace = True) 
        cuit_empresas['letra3'].fillna(cuit_empresas['letra1'], inplace = True) 

        cuit_clae_letra = cuit_empresas.drop(["padron_contribuyentes","actividad_mectra", "letra_mectra"], axis = 1 , inplace = False)
                
        return cuit_clae_letra
    
def predo_bec_bk(bec, bec_to_clae):
    bec_cap = bec[bec["BEC5EndUse"].str.startswith("CAP", na = False)]
    #partes y accesorios dentro de start with cap
    partes_accesorios  = bec_cap[bec_cap["HS6Desc"].str.contains("part|acces")]   
    partes_accesorios["BEC5EndUse"].value_counts()

    # filtro bienes de capital
    bec_bk = bec_cap.loc[~bec_cap["HS6"].isin( partes_accesorios["HS6"])]
    
    bec_to_clae.drop(bec_to_clae.iloc[:, 2:], axis = 1, inplace = True)
    bec_to_clae = bec_to_clae.rename(columns={"BEC Category": "BEC5Category"})
    bec_to_clae["bec_to_clae"] = bec_to_clae["bec_to_clae"].str.replace(",", "").str.replace(" ", "")  

    filtro = [ "HS4", "HS4Desc", "HS6", "HS6Desc", "BEC5Category","BEC5Specification", "BEC5EndUse",
              "BEC4Code", 'BEC4ENDUSE', 'BEC4INT', 'BEC4CONS', 'BEC4CAP' ]
    
    bec_bk = pd.merge(bec_bk[filtro], bec_to_clae , left_on= "BEC5Category", right_on = "BEC5Category", how= "left")
    
    return bec_bk

def def_join_impo_clae(impo_17, cuit_empresas):
    impo_clae = pd.merge(impo_17, cuit_empresas, left_on = "CUIT_IMPOR", right_on = "cuit", how = "right")
    impo_clae.drop(["cuit", "denominacion"], axis=1, inplace = True)
       
    return impo_clae

def def_join_impo_clae_bec(join_impo_clae, bec_bk):
    impo17_bec_bk = pd.merge(join_impo_clae, bec_bk, how= "left" , left_on = "HS6", right_on= "HS6" )

    # filtramos las impos que no mergearon (no arrancan con CAP)
    impo17_bec_bk = impo17_bec_bk[impo17_bec_bk["HS4"].notnull()]

    return impo17_bec_bk

    

def def_join_impo_clae_bec_bk_comercio(join_impo_clae_bec_bk, comercio):
    comercio2 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = { "vta_vehiculos":"vta_vehiculos2",
                                                                                   "vta_bk": "vta_bk2", "vta_sec": "vta_sec2"})

    comercio3 = comercio.drop(["letra", "clae6_desc"] , axis = 1).rename(columns = { "vta_vehiculos":"vta_vehiculos3",
                                                                                   "vta_bk": "vta_bk3", "vta_sec": "vta_sec3"})

    # join de la matriz con el sector comercio
    ## Comercio 1
    impo17_bec_complete = pd.merge(join_impo_clae_bec_bk, comercio.drop(["letra", "clae6_desc"], axis = 1), 
                             how = "left", left_on = "actividad1", right_on = "clae6")
    
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 

    ## Comercio 2
    impo17_bec_complete = pd.merge(impo17_bec_complete, comercio2, 
                                   how = "left", left_on = "actividad2", right_on = "clae6")
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 

    ## Comercio 3
    impo17_bec_complete = pd.merge(impo17_bec_complete , comercio3 , 
                          how = "left", left_on = "actividad3", right_on = "clae6")
    impo17_bec_complete.drop("clae6", axis=1, inplace = True) 

    return  impo17_bec_complete   

    
# =============================================================================
# ponderaciones
# =============================================================================

def def_contingencia(join_impo_clae_bec_bk_comercio):
    impo_by_product_1 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra1" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra1": "letra"})
    impo_by_product_2 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra2" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra2": "letra"})
    impo_by_product_3 = join_impo_clae_bec_bk_comercio.groupby(["HS6", "letra3" ], as_index = False).agg({'valor': sum}).rename(columns = {"letra3": "letra"})
    
    impo_by_product = impo_by_product_1.append([impo_by_product_2,impo_by_product_3], ignore_index=True)
    table = pd.pivot_table(impo_by_product, values='valor', index=['HS6'], columns=['letra'], aggfunc=np.sum, fill_value=0)
    return table

def def_join_impo_clae_bec_bk_comercio_pond(ncm_act_pond, tabla_contingencia):
    x = ncm_act_pond.copy()
    x["letra1_pond"] = np.nan
    x["letra2_pond"] = np.nan
    x["letra3_pond"] = np.nan
    
    
    return x
    
def def_calc_pond(impo,cont):
    join_final = impo.copy()
    for a in tqdm(range(len(join_final))):
        # cuit= join_final.iloc[a]["CUIT_IMPOR"]
        letra_1= join_final.iloc[a]["letra1"]
        letra_2= join_final.iloc[a]["letra2"]
        letra_3= join_final.iloc[a]["letra3"]
        # print(cuit, letra_1, letra_2, letra_3)
        
        x=[]
        for b in [letra_1, letra_2, letra_3]:
            ncm = join_final.iloc[a]["HS6"]
            ncm_val = cont.loc[ncm][b]
            x.append(ncm_val)
        
        total=x[0]+x[1]+x[2]
        act1_pond=x[0]/total
        act2_pond=x[1]/total
        act3_pond=x[2]/total
        join_final.at[a, "letra1_pond"] = act1_pond
        join_final.at[a, "letra2_pond"] = act2_pond
        join_final.at[a, "letra3_pond"] = act3_pond
        # print(ncm, x, total, act1_pond, act2_pond, act3_pond)
    return join_final


# =============================================================================
# matriz
# =============================================================================

def def_insumo_matriz( raw_data):
    
    #creamos df para guardar los insumos de la matriz
    for_fill = pd.DataFrame()
    for_fill ["cuit"]=""
    for_fill ["hs6"]=""
    for_fill ["valor_pond"]=""    
    for_fill ["si"]=""    
    for_fill ["sd"]=""
    for_fill ["ue_dest"]=""

    for a in tqdm(range(len(raw_data))):
    # for a in tqdm(range(1,100)):
        for b, c, d, e in zip(["letra1", "letra2", "letra3"],
                              ["vta_bk", "vta_bk2", "vta_bk3"],
                              ["vta_sec","vta_sec2", "vta_sec3"],
                              ["letra1_pond", "letra2_pond", "letra3_pond"]):
            
            if raw_data.iloc[a][b] == "G":
                if raw_data.iloc[a][c] == 0:
                    letra_sd= raw_data.iloc[a][b]
                    
                elif raw_data.iloc[a][d]==0:
                     letra_sd = "CONS"
                     
                else:
                   letra_sd = None #np.nan #asigno por probabilidad
                       
            else:
                letra_sd = raw_data.iloc[a][b]
               
            
            values = {'cuit': raw_data.iloc[a]["CUIT_IMPOR"],
                      "hs6": raw_data.iloc[a]["HS6"],
                      "valor_pond":raw_data.iloc[a]["valor"]*raw_data.iloc[a][e] ,
                      "si": raw_data.iloc[a]["letra1"],
                      "sd": letra_sd,
                      "ue_dest": "nan"}
            # print (values)
            for_fill= for_fill.append(values, ignore_index=True)
            # print (a/len(raw_data),"%")
    return for_fill
    
def def_matriz_c_prob(prob):
    
    calc_prob = prob.copy()
    calc_none = prob.copy()
    sisd_final = prob.copy()
    
   
    calc_prob = calc_prob[calc_prob["sd"].notnull()]
    calc_prob = calc_prob[calc_prob["sd"]!="G"].groupby(["hs6","sd"])["valor_pond"].agg("sum")
    calc_prob = calc_prob.groupby(level=0).apply(lambda x: x/float(x.sum())).reset_index().rename(columns={"valor_pond":"valor_prob"})
   
    
    calc_none = calc_none[calc_none["sd"].isnull()]
    calc_none = pd.merge(left=calc_none.drop("sd", axis=1), right=calc_prob, left_on="hs6", right_on = "hs6", how="left")
    calc_none["valor_pond"] = calc_none["valor_pond"] * calc_none["valor_prob"] 
    calc_none.drop("valor_prob", axis=1, inplace=True)
        
    matriz_sisd_final = pd.concat([sisd_final[sisd_final["sd"].notnull()], calc_none])
    
    return matriz_sisd_final
                   
    
def to_matriz(matriz_sisd_final):
    z = pd.pivot_table(matriz_sisd_final, values='valor_pond', index=['si'], columns=['sd'], aggfunc=np.sum, fill_value=0)
    cols=list(z.columns.values)
    cols.pop(cols.index("CONS"))
    z=z[cols+["CONS"]] #ubicacion del consumo ultima colummna
    
    z= z.append(pd.Series(name='T')) #imputacion de T
    z= z.replace(np.nan,0)
    
    z= z.append(pd.Series(name='CONS')) #imputacion de CONS
    z= z.replace(np.nan,0)
    
    return z


# =============================================================================
# pre visualizacion
# =============================================================================

def sectores():
    sectores_desc = {"A":	"Agricultura",  "B":	"Minas y canteras", "C":	"Industria", "D": "Energía",
                     "E":	"Agua y residuos", "F":	"Construcción", "G": "Comercio", "H":	"Transporte",
                     "I":	"Alojamiento", "J":	"Comunicaciones", "K":	"Serv. financieros","L":	"Serv. inmobiliarios",
                     "M":	"Serv. profesionales", "N":	"Serv. apoyo", "O":	"Sector público", "P":	"Enseñanza",
                     "Q":	"Serv. sociales", "R":	"Serv. culturales", "S":	"Serv. personales", "T":	"Serv. doméstico",
                     "U": "Serv. organizaciones" , "CONS": "Consumo" }
    return sectores_desc


def impo_total(z, sectores_desc):

    # transformacion a array de np
    z_visual = z.to_numpy()
    
    #diagonal y totales col y row
    col_sum  = np.nansum(z_visual , axis=0)
    
    # sectores
    sectores_desc.pop("U")
    sectores = pd.DataFrame( { "desc":sectores_desc.values(), "letra": sectores_desc.keys() })

    #importaciones totales (ordenadas)
    impo_tot_sec = pd.DataFrame({"impo_tot": col_sum, "letra":sectores_desc.keys() }, index=sectores_desc.values())  
    impo_tot_sec.sort_values(by = "impo_tot", ascending= False, inplace = True)
    
    return impo_tot_sec

def impo_comercio_y_propio(z, sectores_desc):
    # transformacion a array de np
    z_visual = z.to_numpy()
    
    #diagonal y totales col y row
    diagonal = np.diag(z_visual)
    col_sum  = np.nansum(z_visual , axis=0)
    
    #diagonal sobre total col y comercio sobre total
    diag_total_col = diagonal/col_sum
    g_total_col = z_visual[6][:]/col_sum
    comercio_y_propio = pd.DataFrame({"Propio": diag_total_col*100 , 'Comercio': g_total_col*100} , index = sectores_desc.values())
    return comercio_y_propio.sort_values(by = 'Propio', ascending = False) 
    

def top_n(matriz_sisd_final, letras, bec, impo_tot_sec, n):

    x = matriz_sisd_final.groupby(["hs6", "sd"], as_index=False)['valor_pond'].sum("valor_pond")
    top_5_impo = x.reset_index(drop = True).sort_values(by = ["sd", "valor_pond"],ascending = False)
    top_5_impo  = top_5_impo.groupby(["sd"], as_index = True).head(n)
    top_5_impo  = pd.merge(left=top_5_impo, right=letras, left_on="sd", right_on="letra", how="left").drop(["sd"], axis=1)
    top_5_impo  = pd.merge(left=top_5_impo, right=bec[["HS6","HS6Desc"]], left_on="hs6", right_on="HS6", how="left").drop("HS6", axis=1)
    top_5_impo  = pd.merge(top_5_impo  , impo_tot_sec, left_on="letra", right_on="letra", how = "left")
    top_5_impo["impo_relativa"] = top_5_impo["valor_pond"]/top_5_impo["impo_tot"] 
    top_5_impo["short_name"] = top_5_impo["HS6Desc"].str.slice(0,15)

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