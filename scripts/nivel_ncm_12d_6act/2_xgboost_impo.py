# -*- coding: utf-8 -*-
"""xgboost_impo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16uujRCNtIfMg4ZoV-JQxtFRMEqnn3l4I
"""

import os
# os.chdir("C:/Users/Administrator/Documents/equipo investigacion/impo_sectorial/scripts/nivel_ncm_12d_6act")
os.chdir("C:/Archivos/repos/impo_sectorial/scripts/nivel_ncm_12d_6act")
# os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score#, scorer
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error
from urllib.request import urlretrieve
import xgboost as xgb
# from xgboost import XGBClassifier
import scipy.stats.distributions as dists



"""# Datos"""

data_pre = pd.read_csv( "../data/resultados/data_train_test.csv")
data_pre.info()

data_2pred = pd.read_csv("../data/resultados/data_to_pred.csv")
data_2pred.info()

data_model = pd.read_csv("../data/resultados/data_modelo_diaria.csv")

data_model.info()


# data_pre=data_pre.sample( n = 1000, random_state = 42 )
data_pre.shape

#reviso missings
data_pre.isna().sum()
data_pre.dropna(axis = 0, inplace = True)


#reviso y reemplazo infinitos
np.isinf(data_pre).values.sum()
data_pre.replace([np.inf, -np.inf], np.nan, inplace=True)

# frecuencia de la clase
data_pre["bk_dummy"].value_counts(normalize=True)

#split train test
X_train , X_test, y_train, y_test = train_test_split(data_pre.drop("bk_dummy", axis =1),  data_pre["bk_dummy"], test_size = 0.3, random_state = 3 , stratify=data_pre["bk_dummy"])

## Chequeamos el balanceo en los dataset´s
for split_name, split in zip(['Entrenamiento', 'Prueba'],[y_train,y_test]):
  print(split_name, "\n", pd.Series(split).value_counts(normalize=True) , "\n" )
  print(split_name, "\n", pd.Series(split).value_counts(normalize= False),"\n",
        "muestras {}".format(len(split)), "\n")

print("muestras totales {}".format(len(X_train)+len(X_test)))



"""## Modelo de base"""
classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=42, enable_categorical = False)

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


"""## Random y Grid Search """


classifier = xgb.sklearn.XGBClassifier(nthread=-1, objective= 'binary:logistic', seed=42, enable_categorical = False)

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
        'n_estimators': range(10, 100, 1),
        }


random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 5,                        
    verbose=True)

start = datetime.datetime.now()
random_search.fit(X_train, y_train)
end = datetime.datetime.now()
print(end-start)

#mejor modelo
best_xgb = random_search.best_estimator_
best_xgb

#prediccion default (0.5)
y_pred = best_xgb.predict(X_test)
pd.DataFrame(y_pred, index=X_test.index, columns=['bk_dummy']).value_counts()
# plt.hist( y_pred["bk_dummy"])
roc_auc_score(y_test,y_pred)
confusion_matrix(y_test, y_pred, normalize= "pred")#, labels= [1, 0 ])
print(classification_report(y_test, y_pred) )

""" ## Determinacion punto de corte """
#predicciones
y_pred = best_xgb.predict_proba(X_test)
y_pred_df = pd.DataFrame(y_pred, index=X_test.index, columns=["CI", "BK"])

# predicción default
y_pred_default =  np.where(y_pred_df ['BK'] > y_pred_df['CI'], 1, 0)
roc_auc_score(y_test,y_pred_default )

#buscando punto de corte
metrics= pd.DataFrame()
for corte in np.linspace(0.01,0.99, num = 100): # num indica la cantidad de ptos de corte a explorar
    prediccion = np.where(y_pred_df ['BK'] > corte, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, prediccion).ravel()
    dic = {
        "punto_corte":  corte,
        "tp": tp, "tn": tn, "fp" : fp, "fn": fn,
        "pp" : tp+fp, "np": tn+fn,
        "acc" : accuracy_score(y_test, prediccion),
        "precision" : precision_score(y_test, prediccion), #PPV = TP/Predicted Positive
        "recall" : recall_score(y_test, prediccion), #TPR = TP/Real Positive (sensitivity)
        "FPR" : fp/(tn+fp), #"falsa alarma": qué tanto clasificamos positivo cuando el verdadero resultado es negativo
        "auc" : roc_auc_score(y_test, prediccion) 
        }
    metrics= pd.concat([metrics, pd.DataFrame(dic, index = [round(corte,2).astype(str)])])

punto_optimo = metrics.sort_values("auc", ascending=False)["punto_corte"][0]


# grafico curva ROC
plt.plot(metrics["FPR"], metrics["recall"])
plt.plot(0.0769635,0.931076, "ro" )
plt.title("Curva ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")


#violin destribuciones de observaciones (N)
melt_data = pd.melt(metrics, id_vars = "punto_corte", value_vars = ["pp","np"], var_name="ue_dest", value_name="n").sort_values("punto_corte")
ax = sns.violinplot(data = melt_data,  y = "n", x = "ue_dest" )
ax.set_title("Distribuciones de BK y CI para todos los puntos de corte")
ax.set_xlabel("Uso Económico")
ax.set_ylabel("Observaciones")
ax.axhline(200034)
plt.show()

#violin distribucion de probabilidades 
violin_data = pd.DataFrame({"prob_bk": y_pred_df["BK"], "bk_dummy": y_test})
violin_data["bk_dummy"] = np.where(violin_data["bk_dummy"]==1, "BK", "CI") 
ax = sns.violinplot(data = violin_data,  y = "prob_bk", x = "bk_dummy" )
ax.set_title("Distribuciones de probabilidades de datos test (puntos de corte)")
ax.set_xlabel("Uso Economico Observado")
ax.set_ylabel("Probabilidad predicha")
ax.axhline(punto_optimo)
plt.show()

# calibracion del modelo https://www.cienciadedatos.net/documentos/py11-calibrar-modelos-machine-learning.html


"""## Entrenamiento con todos los datos"""

xgb_all = xgb.sklearn.XGBClassifier(base_score=0.5, booster='gbtree',
              colsample_bylevel=0.9176911274292641, colsample_bynode=1,
              colsample_bytree=0.2362802135886268, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.4089751175210069, max_delta_step=0, max_depth=17,
              min_child_weight=1,  monotone_constraints='()',
              n_estimators=54, n_jobs=16, nthread=-1, num_parallel_tree=1,
              random_state=42, reg_alpha=34.93982083457318,
              reg_lambda=28.683948734756893, scale_pos_weight=1, seed=42,
              silent=False, subsample=0.7536819429003822, tree_method='exact',
              validate_parameters=1, verbosity=None, enable_categorical=False)

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

# Modelos
pickle.dump(best_xgb, open('modelos\\xgboost_train_cv.sav', 'wb'))
#best_xgb = pickle.load(open('modelos\\xgboost_train_cv.sav', 'rb'))

pickle.dump(xgb_all, open('modelos\\xgboost_all_data.sav', 'wb'))
#xgb_all = pickle.load(open('modelos\\xgboost_all_data.sav', 'rb'))

plt.figure(figsize=(20,15))
xgb.plot_importance(xgb_all, ax=plt.gca())

###############################################
# datos a predecir
#################################
clasificacion = xgb_all.predict_proba(data_2pred)

clasificacion_df = pd.DataFrame(clasificacion, index=data_2pred.index , columns= ["prob_CI", "prob_BK"])
clasificacion_df["ue_dest"]  = np.where(clasificacion_df["prob_BK"] > punto_optimo, 1, 0)
clasificacion_df["ue_dest"].value_counts()

# clasificacion_prob = xgb_all.predict_proba(data_2pred)
# clasificacion_prob_df = pd.DataFrame(clasificacion_prob)#, columns= ["bk_dummy_0", "bk_dummy_1"])
# clasificacion_df.value_counts()

datos_predichos = data_model[data_model ["ue_dest"] == "?" ]
datos_predichos["bk_dummy"] = clasificacion_df["ue_dest"] 
datos_predichos.to_csv("../data/resultados/datos_clasificados_modelo_all_data.csv", index= False, sep = ";")
# datos_predichos = pd.read_csv("../data/resultados/datos_clasificados_modelo_all_data.csv")

plt.hist(x = "bk_dummy", data = datos_predichos )

for boolean , text in zip([True, False], ["Frecuencias Relativas", "Frecuencias Abosolutas"] ):
  print(text+"\n", datos_predichos.bk_dummy.value_counts(normalize= boolean), "\n" )


# Exportacion de resultados
# from sklearn.externals import joblib

best_xgb.score(X_test, y_test)

y_pred_ = best_xgb.predict(X_test)
roc_auc_score(y_test,y_pred_)
confusion_matrix(y_test, y_pred_, normalize= "pred")#, labels= [1, 0 ])
cm = confusion_matrix(y_test, y_pred_)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_).ravel()
(tn, fp, fn, tp )
plot_confusion_matrix(best_xgb, X_test, y_test)                                  

import seaborn as sns
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Etiqueta predicha');ax.set_ylabel('Etiqueta verdadera'); 
ax.set_title('Matriz de confusión con datos de test'); 
ax.xaxis.set_ticklabels(['CI', 'BK']); ax.yaxis.set_ticklabels(['CI', 'BK']);

print(classification_report(y_test, y_pred_) )

# joblib.dump(xgboos_rscv_all, 'knn_all_data.pkl')
# joblib.dump(best_xgb, 'knn_clasif_data.pkl')

# Clasficacion de Xgboost entrenado con todos los datos
# clasificacion_df.to_csv("../data/resultados/datos_clasificados_modelo_train.csv")
# clasificacion_df.to_csv("../data/resultados/datos_clasificados_modelo_all_data.csv")

# joblib.load("knn_gscv_all.pkl")


