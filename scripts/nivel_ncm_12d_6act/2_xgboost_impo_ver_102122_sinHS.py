# =============================================================================
# Prueba sin HS
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
data_pre.drop(["HS6", "HS8","HS10"], axis =1, inplace = True)
data_2pred.drop(["HS6", "HS8","HS10"], axis =1, inplace = True)

#reviso y reemplazo infinitos
np.isinf(data_pre).values.sum()
data_pre.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train , X_test, y_train, y_test = train_test_split(data_pre.drop("bk_dummy", axis =1),  data_pre["bk_dummy"], test_size = 0.3, random_state = semilla )#, stratify=data_pre["bk_dummy"])


classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=semilla)

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

#mejores parámetros
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


#############################
# Métricas de test
##############################
y_pred_ = best_xgb.predict_proba(X_test)
y_pred_df = pd.DataFrame( y_pred_, index=X_test.index , columns= [ "prob_CI", "prob_BK"])
y_pred_df["y_test"] = y_test
y_pred_df["y_pred"]  = np.where(y_pred_df["prob_BK"] > punto_optimo, 1, 0)
y_pred_df["valor"] = X_test.valor


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
