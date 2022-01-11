# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:29:24 2022

@author: mateo
"""

import pandas as pd 

train = pd.read_csv("C:/Users/mateo/Downloads/validation.csv")
validation = pd.read_csv("C:/Users/mateo/Downloads/validation.csv")

train.shape
validation.shape


#
datos_vm = pd.read_csv("../data/heavys/datos_clasificados_modelo_all_data.csv", sep = ";") 
datos_sm = pd.read_csv("../data/heavys/SM_datos_clasificados_modelo_all_data.csv", sep = ";") 


datos_vm["ue_dest_sm"] = datos_sm["ue_dest"]
