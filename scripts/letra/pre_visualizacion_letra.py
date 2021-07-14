# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:34:56 2021

@author: mateo
"""


import pandas as pd

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