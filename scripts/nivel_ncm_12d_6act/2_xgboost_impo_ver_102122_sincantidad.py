# =============================================================================
# Prueba sin cantidad
# =============================================================================
import os
# os.chdir("C:/Users/Administrator/Documents/equipo investigacion/impo_sectorial/scripts/nivel_ncm_12d_6act")
# os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import  json
import datetime
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score,  classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, mean_absolute_error
from urllib.request import urlretrieve
import xgboost as xgb
# from xgboost import XGBClassifier
import scipy.stats.distributions as dists

##
semilla = 42

"""# Datos"""
data_pre = pd.read_csv( "../data/heavys/data_train_test_21oct.csv") #data_pre y data_2pred posee los datos como los necesita el modelo
data_2pred = pd.read_csv("../data/resultados/data_to_pred_21oct.csv")
data_model = pd.read_csv("../data/heavys/data_modelo_diaria.csv")  #data_model posee los datos que se utilizaran en el script 3


#pruebo borrar las columnas 
data_pre.drop(["cant_est", "cant_decl"], axis =1, inplace = True)
data_2pred.drop(["cant_est", "cant_decl"], axis =1, inplace = True)
