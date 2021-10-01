# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:34:06 2021

@author: mateo
"""

# =============================================================================
# LIBRERIAS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.model_selection import  StratifiedShuffleSplit, KFold, GridSearchCV, RandomizedSearchCV #, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report #accuracy_score, precision_score, recall_score, f1_score, make_scorer

import xgboost as xgb

from Bases_nivel_ncm_12d_6act import *
# from scipy.stats import *

#visto en AA
# from sklearn.neighbors import KNeighborsClassifier 

#visto en cuml example
# from sklearn.neighbors import NearestNeighbors as skKNN


# =============================================================================
# DATA
# =============================================================================

data= pd.read_csv("../data/resultados/data_modelo.csv")

data.info()

#preprocesamiento etiquetados
cols =  [ "HS6",  
         'valor',  'kilos', "precio_kilo" , 
         "letra1",	"letra2",	"letra3", 	"letra4", 	"letra5",	"letra6"	,
         "uni_est",	"cant_est",	"uni_decl",	"cant_decl",  
         "metric" ,
         "ue_dest"]

data = data[cols]

# datos etiquetados
data_clasif = data[data["ue_dest"] != "?" ]


# dummy de BK
data_clasif['HS6'] = data_clasif['HS6'].astype("str")
data_clasif["bk_dummy"] = data_clasif["ue_dest"].map({"BK": 1, "CI": 0})
data_clasif.drop("ue_dest", axis = 1, inplace = True)

#Filtros de columnas
cat_col = data_clasif.select_dtypes(include=['object']).columns
num_col = data_clasif.select_dtypes(include=['float', "int64" ]).columns


data_pre = pd.concat( [ str_a_num(data_clasif[cat_col]) , data_clasif[num_col] ], axis = 1  )

# datos no etiquetados
data_not_clasif = data[data["ue_dest"] == "?" ]
data_not_clasif['HS6'] = data_not_clasif['HS6'].astype("str")
# data_not_clasif["bk_dummy"] = data_not_clasif["ue_dest"].map({"BK": 1, "CI": 0})
data_not_clasif.drop("ue_dest", axis = 1, inplace = True)

data_2_pred = pd.concat( [ str_a_num(data_not_clasif[cat_col]) , data_not_clasif[['valor', 'kilos', 'precio_kilo', 'cant_est', 'cant_decl', 'metric']] ], axis = 1  )

data_2_pred.to_csv("../data/resultados/data_xgb_to_pred.csv", index=False)
data_pre.to_csv("../data/resultados/data_xgb_train.csv", index=False)

# =============================================================================
# Armado del modelo
# =============================================================================

X_train , X_test, y_train, y_test = train_test_split(data_pre.drop("bk_dummy", axis =1), 
                                                     data_pre["bk_dummy"], 
                                                     test_size = 0.3, random_state = 3)

from xgboost import XGBClassifier


clf = xgb.XGBClassifier()

param_grid = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}

fit_params = {'eval_metric': 'mlogloss',
              'early_stopping_rounds': 10,
              'eval_set': [(x_valid, y_valid)]}

rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=20,
                            n_jobs=1, verbose=2, cv=2,
                            fit_params=fit_params,
                            scoring='neg_log_loss', refit=False, random_state=42)