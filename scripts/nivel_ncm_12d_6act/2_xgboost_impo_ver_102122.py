
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score#, scorer
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error
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

data_model.ue_dest.value_counts()

data_pre.info()
data_2pred.info()
data_model.info()

#sampleo
# data_pre=data_pre.sample( n = 1000, random_state = semilla )

###########################
##  Preprocesamiento     ##
###########################

#reviso missings
data_pre.isna().sum().sum() #no hay missings
# data_pre.dropna(axis = 0, inplace = True)

#reviso y reemplazo infinitos
np.isinf(data_pre).values.sum()
data_pre.replace([np.inf, -np.inf], np.nan, inplace=True)


# Prueba con todos los datos
#########################
# Split train test
###########################
# frecuencia de la clase
data_pre["bk_dummy"].value_counts(normalize=True)

#split train test
X_train , X_test, y_train, y_test = train_test_split(data_pre.drop("bk_dummy", axis =1),  data_pre["bk_dummy"], test_size = 0.3, random_state = semilla )#, stratify=data_pre["bk_dummy"])

## Chequeamos el balanceo en los dataset´s
for split_name, split in zip(['Entrenamiento', 'Prueba'],[y_train,y_test]):
  print(split_name, "\n", pd.Series(split).value_counts(normalize=True) , "\n" )
  print(split_name, "\n", pd.Series(split).value_counts(normalize= False),"\n",
        "muestras {}".format(len(split)), "\n")

print("muestras totales {}".format(len(X_train)+len(X_test)))
print("registros completos?", (len(X_train)+len(X_test)+len(data_2pred))  == len(data_model) )

#############################
# """## Modelo de base"""
#############################

classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=semilla)#, enable_categorical = False)

start = datetime.datetime.now()
classifier.fit(X_train, y_train)
end = datetime.datetime.now()
print(end-start)

y_pred = classifier.predict(X_test)
pd.DataFrame(y_pred, index=X_test.index, columns=['bk_dummy']).value_counts()

roc_auc_score(y_test,y_pred)
confusion_matrix(y_test, y_pred, normalize= "pred")#, labels= [1, 0 ])
print(classification_report(y_test, y_pred) )

#############################
# """## Random Search """
############################
classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=semilla)#objective= 'binary:logistic',  enable_categorical = False)

# parameters = {'silent': [False],
#         'max_depth':  range(1, 20, 2),
#         "max_leaves": range(0, 1000, 10),
#         'learning_rate': dists.uniform(0.001, 1), # continuous distribution
#         # param2=dists.randint(16, 512 + 1), # discrete distribution
#         # param3=['foo', 'bar'],             # specifying possible values directly
#         "min_child_weight": range(1, 50, 2),
#         "max_bin": range(5, 120, 5),
#         'subsample': dists.uniform(0.001, 0.999) ,
#         'colsample_bytree': dists.uniform(0.001, 0.999),
#         'colsample_bylevel': dists.uniform(0.001, 0.999),
#         'reg_lambda':dists.uniform(1, 100),
#         'reg_alpha': dists.uniform(1, 200),
#         'n_estimators': range(50, 200, 1),
#         "tree_method": ['hist'],
#         "grow_policy": ['lossguide']
#         }

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

random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=parameters,
    scoring = 'roc_auc',
    random_state= semilla,
    n_jobs = -1,
    cv = 10,
     n_iter=150,
    verbose=True)

start = datetime.datetime.now()
random_search.fit(X_train, y_train)
end = datetime.datetime.now()
print(end-start)

#cv 's
cv_results = pd.DataFrame(random_search.cv_results_)
cv_results.to_csv("./modelos/random_cv_results_21oct_100iters.csv")
cv_results.info()

# plt.hist(cv_results["mean_test_score"])
# plt.hist(cv_results["std_test_score"])

params_stables = cv_results.sort_values("std_test_score")["params"][0]

mejores_parametros = random_search.best_params_
mejores_parametros

#mejor modelo
best_xgb = random_search.best_estimator_
best_xgb

################################
# Determinacion punto de corte 
############################
#predicciones default
y_pred = best_xgb.predict_proba(X_test)
y_pred_df = pd.DataFrame(y_pred, index=X_test.index, columns=["CI", "BK"])

# predicción default
y_pred_default =  np.where(y_pred_df ['BK'] > y_pred_df['CI'], 1, 0)
roc_auc_score(y_test,y_pred_df ['BK'] )
confusion_matrix(y_test,y_pred_default, normalize= "pred")#, labels= [1, 0 ])
print(classification_report(y_test, y_pred_default) )

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
fpr_op = metrics.sort_values("auc", ascending=False)["FPR"][0]
recall_op =metrics.sort_values("auc", ascending=False)["recall"][0]

# # grafico curva ROC
# plt.plot(metrics["FPR"], metrics["recall"])
# plt.plot(fpr_op,recall_op, "ro" )
# plt.title("Curva ROC")
# plt.xlabel("FPR")
# plt.ylabel("TPR")
#
# #violin destribuciones de observaciones (N)
# melt_data = pd.melt(metrics, id_vars = "punto_corte", value_vars = ["pp","np"], var_name="ue_dest", value_name="n").sort_values("punto_corte")
# # ax = sns.violinplot(data = melt_data,  y = "n", x = "ue_dest" )
# # ax.set_title("Distribuciones de BK y CI para todos los puntos de corte")
# # ax.set_xlabel("Uso Económico")
# # ax.set_ylabel("Observaciones")
# # # ax.axhline(200034)
# # plt.show()

# #violin distribucion de probabilidades
# violin_data = pd.DataFrame({"prob_bk": y_pred_df["BK"], "bk_dummy": y_test})
# violin_data["bk_dummy"] = np.where(violin_data["bk_dummy"]==1, "BK", "CI")
# ax = sns.violinplot(data = violin_data,  y = "prob_bk", x = "bk_dummy" )
# ax.set_title("Distribuciones de probabilidades de datos test (puntos de corte)")
# ax.set_xlabel("Uso Economico Observado")
# ax.set_ylabel("Probabilidad predicha")
# ax.axhline(punto_optimo)
# plt.show()

# calibracion del modelo https://www.cienciadedatos.net/documentos/py11-calibrar-modelos-machine-learning.html

#############################
# Métricas de test
##############################
y_pred_ = best_xgb.predict_proba(X_test)
y_pred_df = pd.DataFrame( y_pred_, index=X_test.index , columns= [ "prob_CI", "prob_BK"])
y_pred_df["y_test"] = y_test
y_pred_df["y_pred"]  = np.where(y_pred_df["prob_BK"] > punto_optimo, 1, 0)
y_pred_df["valor"] = X_test.valor

#sns.heatmap(confusion_matrix(y_test, y_pred), annot = True , fmt = ".0f",
#            xticklabels = ["CI", "BK"], yticklabels = ["CI", "BK"])
#plt.title("Matriz de confusión. Datos de test")

# Confussion matrix in USD
y_pred_df["error"] = np.where(y_pred_df["y_pred"]!=y_pred_df["y_test"] , y_pred_df["valor"],0)
y_pred_df["error"].sum()/y_pred_df["valor"].sum()*100

y_pred_df["clasificacion"] =np.where((y_pred_df["y_pred"]==1) & (y_pred_df["y_test"]==1), "TP",
                                     np.where((y_pred_df["y_pred"]==1) & (y_pred_df["y_test"]==0), "FP", 
                                              np.where((y_pred_df["y_pred"]==0) & (y_pred_df["y_test"]==0), "TN",
                                                       np.where((y_pred_df["y_pred"]==0) & (y_pred_df["y_test"]==1), "FN", np.nan
                                                                )
                                                       ) 
                                              )
                                     )
cm_usd = y_pred_df.groupby(["clasificacion"])["valor"].sum()/1e6
cm_usd_plt = pd.DataFrame({ "CI": cm_usd.iloc[[2,1]].values, "BK": cm_usd.values[0::3]} , index= ["CI", "BK"]   )  
#sns.heatmap(cm_usd_plt, annot=True, fmt=".0f")
#plt.title("Matriz de confusión en millones de USD. Datos de test")


#metricas
y_pred = y_pred_df["y_pred"]
roc_auc_score(y_test,y_pred )
confusion_matrix(y_test, y_pred , normalize= "pred")#, labels= [1, 0 ])
cm = confusion_matrix(y_test, y_pred )
tn, fp, fn, tp = confusion_matrix(y_test, y_pred ).ravel()
(tn, fp, fn, tp )

# neg: CI; pos: BK
y_pred.value_counts()
y_test.value_counts()
tn+fp
tp+fn



#ax= plt.subplot()
#sns.heatmap(cm, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
#ax.set_xlabel('Etiqueta predicha');ax.set_ylabel('Etiqueta verdadera');
#ax.set_title('Matriz de confusión con datos de test');
#ax.xaxis.set_ticklabels(['CI', 'BK']); ax.yaxis.set_ticklabels(['CI', 'BK']);

#print(classification_report(y_test, y_pred_df["ue_dest"]) )

###########################################
#"""## Entrenamiento con todos los datos"""
###########################################
best_xgb.get

xgb_all = xgb.sklearn.XGBClassifier(**mejores_parametros, seed=semilla)
# xgb_all = xgb.sklearn.XGBClassifier(**random_search.best_params_, seed=semilla)

start = datetime.datetime.now()
xgb_all.fit(data_pre.drop("bk_dummy", axis =1),  data_pre["bk_dummy"])
end = datetime.datetime.now()
print(end-start)

# plt.figure(figsize=(20,15))
# xgb.plot_importance(xgb_all, ax=plt.gca())

#######################################
# """### Exportacion de los modelos"""
#######################################
# Modelos
pickle.dump(best_xgb, open('modelos\\xgboost_train_cv_21oct_100iters.sav', 'wb')) #guarda el modelo
best_xgb = pickle.load(open('modelos\\xgboost_train_cv_21oct.sav', 'rb')) #carga

pickle.dump(xgb_all, open('modelos\\xgboost_all_data_21oct_100iters.sav', 'wb'))
xgb_all = pickle.load(open('modelos\\xgboost_all_data_21oct.sav', 'rb'))

with open('modelos\\mejores_parametros_150iters.json', 'w') as fp:
    json.dump(mejores_parametros, fp)
with open('modelos\\mejores_estables_150iters.json', 'w') as fp:
    json.dump(params_stables, fp)

###############################################
# Predección de nuevas observaciones
################################################
punto_optimo  =0.6
clasificacion = xgb_all.predict_proba(data_2pred)
clasificacion_df = pd.DataFrame(clasificacion, index=data_2pred.index , columns= ["prob_CI", "prob_BK"])
clasificacion_df["ue_dest"]  = np.where(clasificacion_df["prob_BK"] > punto_optimo, "BK", "CI")
clasificacion_df["ue_dest"].value_counts()

datos_predichos = data_model[data_model ["ue_dest"] == "?" ] #data model posee todos los datos
datos_predichos["ue_dest"] = clasificacion_df["ue_dest"]
datos_predichos["prob_bk"] = clasificacion_df["prob_BK"]

from def_bases_nivel_ncm_12d_6act import *
ncm12_desc = pd.read_csv("../data/d12_2012-2017.csv", sep=";")
ncm12_desc_mod = predo_ncm12_desc(ncm12_desc )

datos_predichos= pd.merge(datos_predichos, ncm12_desc_mod, left_on = "HS6_d12", right_on ="HS_12d", how = "left")

datos_predichos.to_csv("../data/resultados/predicciones_model_21oct.csv" , sep = ";") 

#plt.hist(x = "ue_dest", data = datos_predichos )

for boolean , text in zip([True, False], ["Frecuencias Relativas", "Frecuencias Abosolutas"] ):
  print(text+"\n", datos_predichos.ue_dest.value_counts(normalize= boolean), "\n" )



#############################
# Exportacion de resultados
#############################

datos_all = pd.concat([datos_predichos , data_model[data_model ["ue_dest"] != "?" ] ] , axis = 0) 
len(data_model )== len(datos_all)

datos_all.to_csv("../data/heavys/datos_clasificados_modelo_all_data_21oct.csv", index= False, sep = ";")
# datos_predichos = pd.read_csv("../data/resultados/datos_clasificados_modelo_all_data.csv")


# =============================================================================
# Prueba sin HS
# =============================================================================
#pruebo borrar las columnas de cantidad
data_pre.drop(["cant_est", "cant_decl"], axis =1, inplace = True)
data_2pred.drop(["cant_est", "cant_decl"], axis =1, inplace = True)


X_train , X_test, y_train, y_test = train_test_split(data_pre.drop("bk_dummy", axis =1),  data_pre["bk_dummy"], test_size = 0.3, random_state = semilla )#, stratify=data_pre["bk_dummy"])

#Revisar si es necesario pisar estas variables
# classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=semilla)#objective= 'binary:logistic',  enable_categorical = False)

# random_search = RandomizedSearchCV(
#     estimator=classifier,
#     param_distributions=parameters,
#     scoring = 'roc_auc',
#     random_state= semilla,
#     n_jobs = -1,
#     cv = 10,
#      n_iter=150,
#     verbose=True)

start = datetime.datetime.now()
random_search.fit(X_train, y_train)
end = datetime.datetime.now()
print(end-start)

#mejor modelo
best_xgb = random_search.best_estimator_
best_xgb


#############################
# Métricas de test
##############################
y_pred_ = best_xgb.predict_proba(X_test)
y_pred_df = pd.DataFrame( y_pred_, index=X_test.index , columns= [ "prob_CI", "prob_BK"])
y_pred_df["y_test"] = y_test
y_pred_df["y_pred"]  = np.where(y_pred_df["prob_BK"] > punto_optimo, 1, 0)
y_pred_df["valor"] = X_test.valor








