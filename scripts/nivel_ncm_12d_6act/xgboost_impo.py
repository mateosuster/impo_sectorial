# -*- coding: utf-8 -*-
"""xgboost_impo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16uujRCNtIfMg4ZoV-JQxtFRMEqnn3l4I

# Librerias
"""
import os
os.chdir("C:/Users/Administrator/Documents/equipo investigacion/impo_sectorial/scripts/nivel_ncm_12d_6act")
# os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score#, scorer
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error

from urllib.request import urlretrieve

import xgboost as xgb
# from xgboost import XGBClassifier

import scipy.stats.distributions as dists



"""# Datos"""

data_pre = pd.read_csv( "../data/resultados/data_train_test.csv")
data_pre.head()


# data_pre.dropna(inplace=True)

# data_pre.dropna(inplace=True)
# data_pre=data_pre.sample( n = 1000, random_state = 42 )
data_pre.shape
data_pre.isna().sum()
np.isinf(data_pre).values.sum()

data_pre.replace([np.inf, -np.inf], np.nan, inplace=True)


data_pre["bk_dummy"].value_counts(normalize=True)

X_train , X_test, y_train, y_test = train_test_split(data_pre.drop("bk_dummy", axis =1),  data_pre["bk_dummy"], test_size = 0.3, random_state = 3)#, stratify=True)

## Chequeamos el balanceo en los dataset´s
for split_name, split in zip(['Entrenamiento', 'Prueba'],[y_train,y_test]):
  print(split_name, "\n", pd.Series(split).value_counts(normalize=True) , "\n" )
  print(split_name, "\n", pd.Series(split).value_counts(normalize= False),"\n",
        "muestras {}".format(len(split)), "\n")

print("muestras totales {}".format(len(X_train)+len(X_test)))



"""## Modelo de base"""
classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=42)

start = datetime.datetime.now()
classifier.fit(X_train, y_train)
end = datetime.datetime.now()
print(end-start)

print("Number of boosting trees: {}".format(classifier.n_estimators))
print("Max depth of trees: {}".format(classifier.max_depth))
print("Objective function: {}".format(classifier.objective))

y_pred = classifier.predict(X_test)
pd.DataFrame(y_pred, index=X_test.index, columns=['bk_dummy']).value_counts()

roc_auc_score(y_test,y_pred)

confusion_matrix(y_test, y_pred, normalize= "pred")#, labels= [1, 0 ])

print(classification_report(y_test, y_pred) )

plt.figure(figsize=(20,15))
xgb.plot_importance(classifier, ax=plt.gca())

plt.figure(figsize=(20,15))
xgb.plot_tree(classifier, ax=plt.gca())

"""## Random y Grid Search


"""

classifier = xgb.sklearn.XGBClassifier(nthread=-1, objective= 'binary:logistic', seed=42)

# parameters = {
#     'max_depth': range (2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05]
# }

parameters = {'silent': [False],
        'max_depth':  range(1, 20, 2),
        'learning_rate': dists.uniform(0.01, 1), # continuous distribution
        # param2=dists.randint(16, 512 + 1), # discrete distribution
        # param3=['foo', 'bar'],             # specifying possible values directly
        'subsample': dists.uniform(0.001, 0.999) ,
        'colsample_bytree': dists.uniform(0.001, 0.999),
        'colsample_bylevel': dists.uniform(0.001, 0.999),
        'reg_lambda':dists.uniform(1, 100),
        'reg_alpha': dists.uniform(1, 200),
        'n_estimators': range(10, 100, 1)
        }


# grid_search = GridSearchCV(
#     estimator=classifier,
#     param_grid=parameters,
#     scoring = 'roc_auc',
#     n_jobs = 10,
#     cv = 10,
#     verbose=True
# )
# grid_search.fit(X_train, y_train)

random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,                        
    verbose=True
)

start = datetime.datetime.now()
random_search.fit(X_train, y_train)
end = datetime.datetime.now()
print(end-start)

best_xgb = random_search.best_estimator_
best_xgb


y_pred = best_xgb.predict(X_test)
pd.DataFrame(y_pred, index=X_test.index, columns=['bk_dummy']).value_counts()

# plt.hist( y_pred["bk_dummy"])

roc_auc_score(y_test,y_pred)

confusion_matrix(y_test, y_pred, normalize= "pred")#, labels= [1, 0 ])

print(classification_report(y_test, y_pred) )

plt.figure(figsize=(20,15))
xgb.plot_importance(best_xgb, ax=plt.gca())

"""## Entrenamiento con todos los datos"""

xgb_all = xgb.sklearn.XGBClassifier(base_score=0.5, booster='gbtree',
              colsample_bylevel=0.8503047842259737, colsample_bynode=1,
              colsample_bytree=0.4503608937420998, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.9619399737977476, max_delta_step=0, max_depth=13,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=35, n_jobs=16, nthread=-1, num_parallel_tree=1,
              random_state=42, reg_alpha=62.48882975462377,
              reg_lambda=57.71200766745209, scale_pos_weight=1, seed=42,
              silent=False, subsample=0.33899399224774746, tree_method='exact',
              validate_parameters=1, verbosity=None  )

start = datetime.datetime.now()
xgb_all.fit(X_train, y_train)
end = datetime.datetime.now()
print(end-start)



# xgboos_rscv_all = RandomizedSearchCV(
#     estimator=classifier,
#     param_distributions=parameters,
#     scoring = 'roc_auc',
#     n_jobs = 10,
#     cv = 10,
#     verbose=True)
# xgboos_rscv_all.fit(data_pre.drop("bk_dummy", 1), data_pre["bk_dummy"]   )
# best_xgb = xgboos_rscv_all.best_estimator_
# best_xgb

"""### Predicción de nuevas observaciones"""

data_2pred = pd.read_csv("../data/resultados/data_to_pred.csv")
data_2pred.head()

data_2pred.info()

clasificacion = xgb_all.predict(data_2pred)
clasificacion_df = pd.DataFrame(clasificacion, columns= ["bk_dummy"])
clasificacion_df.value_counts()

data_model = pd.read_csv("../data/resultados/data_modelo_diaria.csv")
datos_predichos = data_model[data_model ["ue_dest"] == "?" ]
datos_predichos["bk_dummy"] = clasificacion
datos_predichos.to_csv("../data/resultados/datos_clasificados_modelo_all_data.csv")

plt.hist(x = "bk_dummy", data = clasificacion_df)

for boolean , text in zip([True, False], ["Frecuencias Relativas", "Frecuencias Abosolutas"] ):
  print(text+"\n", clasificacion_df.bk_dummy.value_counts(normalize= boolean), "\n" )


# Exportacion de resultados
# from sklearn.externals import joblib


# Modelos
pickle.dump(best_xgb, open('xgboost_train_cv.sav', 'wb'))
model = pickle.load(open('xgboost_train_cv.sav', 'rb'))

pickle.dump(xgb_all, open('xgboost_all_data.sav', 'wb'))
model = pickle.load(open('xgboost_all_data.sav', 'rb'))

model.score(X_test, y_test)
y_pred_ = model.predict(X_test)
roc_auc_score(y_test,y_pred_)

# joblib.dump(xgboos_rscv_all, 'knn_all_data.pkl')
# joblib.dump(best_xgb, 'knn_clasif_data.pkl')

# Clasficacion de Xgboost entrenado con todos los datos
# clasificacion_df.to_csv("../data/resultados/datos_clasificados_modelo_train.csv")
# clasificacion_df.to_csv("../data/resultados/datos_clasificados_modelo_all_data.csv")

# joblib.load("knn_gscv_all.pkl")