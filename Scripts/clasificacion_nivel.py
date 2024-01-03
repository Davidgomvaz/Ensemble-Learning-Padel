# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 00:13:36 2023

@author: david
"""

#%% Preparamos los datos de los golpes de padel
from statistics import mode
from sklearn.model_selection import train_test_split
import pandas as pd

# Obtenemos los datos del archivo
datos = pd.read_csv("/Users/david/Desktop/ETSI/4ºCurso/TFg/Golpes/Dataset12.csv")

# Eliminamos las columnas que no nos interesan
datos.drop(columns = ["mano", "reves", "altura", "edad", "sexo", "numero_golpe", "tiempo_golpe"], inplace=True)
X = datos.drop(columns = ["nivel", "tipo_golpe","id"])
nivel = datos["nivel"]
golpe = datos["tipo_golpe"]
iden = datos["id"]
# Dividimos los datos en datos de entramiento (70% de los datos totales) y datos de prueba 30% 

X_train, X_test, y_train, y_test, X_golpe, y_golpe, X_id, y_id = train_test_split(X, nivel,golpe,iden, test_size=0.3, stratify=nivel, random_state=6)
print("La forma de los datos es:")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape,X_golpe.shape, y_golpe.shape, X_id.shape, y_id.shape)
# print(y_test.index)
# print(y_golpe.index)
indices=list(y_golpe.index)
# print(indices)

#%% Definimos el clasificador y mostramos su resultado

from sklearn import model_selection
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from numpy import mean
from numpy import std
import timeit

# Funcion para mostrar la matriz de confusion
niveles = ['Iniciacion','Amateur','Inter','Avanzado   ','Profesional']
def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    plt.xticks(range(5), niveles)
    plt.yticks(range(5), niveles)
    pass

# Funcion para encontrar los mejores parametros
def best_params():
    param_grid={"max_depth": [1, 10, 20, 30, 40],
              "min_samples_split":[2, 4, 8, 10, 20, 100],
              "min_samples_leaf": [1, 2, 3, 4, 5, 6, 10],
              "criterion":["entropy","gini"]}
    model = model_selection.GridSearchCV(estimator= DecisionTreeClassifier(),
                                          param_grid=param_grid,
                                          scoring="accuracy",
                                          cv=5)
    model.fit(X_train, y_train)
    print("val. score: %s" % model.best_score_)
    print("test score: %s" % model.score(X_test, y_test))
    print("Mejores parámetros:", model.best_params_)
    pass

def iter_clf():
    result=list()
    for i in range(10):
        dt_clf = DecisionTreeClassifier(max_depth=30,min_samples_split=2,min_samples_leaf=2,criterion='entropy')
        dt_clf.fit(X_train, y_train)
        y_pred = dt_clf.predict(X_test)
        score=accuracy_score(y_test, y_pred)*100
        print(i, score)
        result.append(score)
    print("Precision: %.3f%% (+/-%.3f)" % (mean(result) , std(result)))
    pyplot.figure()
    pyplot.boxplot(result)
    pyplot.title('Resultado Árbol de Decisión')
    pyplot.ylabel('Precisión(%)')

def tipo_golpe(y_pred):
    identidad_1=list()
    identidad_2=list()
    identidad_3=list()
    identidad_4=list()
    identidad_5=list()
    ini_ama=list()
    ini_inter=list()
    ini_ava=list()
    ini_pro=list()
    ama_ini=list()
    ama_inter=list()
    ama_ava=list()
    ama_pro=list()
    inter_ini=list()
    inter_ama=list()
    inter_ava=list()
    inter_pro=list()
    ava_ini=list()
    ava_inter=list()
    ava_ama=list()
    ava_pro=list()
    pro_ini=list()
    pro_inter=list()
    pro_ava=list()
    pro_ama=list()
    valor = y_pred-y_test.values
    # print(valor)
    for i in range(1216):
        if y_test[indices[i]] ==1:
            if valor[i] == 1:
                ini_ama.append(y_golpe[indices[i]])
                identidad_1.append(y_id[indices[i]])
            elif valor[i] == 2:
                ini_inter.append(y_golpe[indices[i]])
                identidad_2.append(y_id[indices[i]])
            elif valor[i] == 3:
                ini_ava.append(y_golpe[indices[i]])
                identidad_3.append(y_id[indices[i]])
            elif valor[i] == 4:
                ini_pro.append(y_golpe[indices[i]])
                identidad_4.append(y_id[indices[i]])
        
        elif y_test[indices[i]] ==2:
            if valor[i] == -1:
                ama_ini.append(y_golpe[indices[i]])
                # identidad_1.append(y_id[indices[i]])
            elif valor[i] == 1:
                ama_inter.append(y_golpe[indices[i]])
                # identidad_2.append(y_id[indices[i]])
            elif valor[i] == 2:
                ama_ava.append(y_golpe[indices[i]])
                # identidad_3.append(y_id[indices[i]])
            elif valor[i] == 3:
                ama_pro.append(y_golpe[indices[i]])
                # identidad_4.append(y_id[indices[i]])
                
        elif y_test[indices[i]] ==3:
            if valor[i] == -2:
                inter_ini.append(y_golpe[indices[i]])
                # identidad_1.append(y_id[indices[i]])
            elif valor[i] == -1:
                inter_ama.append(y_golpe[indices[i]])
                # identidad_2.append(y_id[indices[i]])
            elif valor[i] == 1:
                inter_ava.append(y_golpe[indices[i]])
                # identidad_3.append(y_id[indices[i]])
            elif valor[i] == 2:
                inter_pro.append(y_golpe[indices[i]])
                # identidad_4.append(y_id[indices[i]])
                
        elif y_test[indices[i]] ==4:
            if valor[i] == -3:
                ava_ini.append(y_golpe[indices[i]])
                # identidad_1.append(y_id[indices[i]])
            elif valor[i] == -2:
                ava_ama.append(y_golpe[indices[i]])
                # identidad_2.append(y_id[indices[i]])
            elif valor[i] == -1:
                ava_inter.append(y_golpe[indices[i]])
                # identidad_3.append(y_id[indices[i]])
            elif valor[i] == 1:
                ava_pro.append(y_golpe[indices[i]])
                # identidad_4.append(y_id[indices[i]])
                
        elif y_test[indices[i]] ==5:
            if valor[i] == -4:
                pro_ini.append(y_golpe[indices[i]])
                # identidad_1.append(y_id[indices[i]])
            elif valor[i] == -3:
                pro_ama.append(y_golpe[indices[i]])
                # identidad_2.append(y_id[indices[i]])
            elif valor[i] == -2:
                pro_inter.append(y_golpe[indices[i]])
                # identidad_3.append(y_id[indices[i]])
            elif valor[i] == -1:
                pro_ava.append(y_golpe[indices[i]])
                # identidad_4.append(y_id[indices[i]])
                
    print("iniciacion")
    print(ini_ama)
    print(ini_inter)
    print(ini_ava)
    print(ini_pro)
    # print("amateur")
    # print(ama_ini)
    # print(ama_inter)
    # print(ama_ava)
    # print(ama_pro)
    # print("intermedio")
    # print(inter_ini)
    # print(inter_ama)
    # print(inter_ava)
    # print(inter_pro)
    # print("avanzado")
    # print(ava_ini)
    # print(ava_ama)
    # print(ava_inter)
    # print(ava_pro)
    # print("profesional")
    # print(pro_ini)
    # print(pro_ama)
    # print(pro_inter)
    # print(pro_ava)
    print("ID")
    print(identidad_1)
    print(identidad_2)
    print(identidad_3)
    print(identidad_4)
    # print(identidad_5)
    pass
        

# Funcion para definir y entrenar el clasificador   
def dt_classifier():
    svm_clf = svm.SVC(C=10, decision_function_shape='ovo', kernel='rbf', probability=True)
    knn_clf = KNeighborsClassifier(n_neighbors=1,p=1)
    dt_clf = DecisionTreeClassifier(max_depth=30,min_samples_split=2,min_samples_leaf=2,criterion='entropy')
    mlp = MLPClassifier(hidden_layer_sizes=(500,250), activation='relu',batch_size=10,max_iter=70, random_state=5)
    clf = VotingClassifier(
    estimators=[('mlp', mlp),('svc', svm_clf), ('dt', dt_clf), ('knn', knn_clf)],
    voting='soft', weights=[3,3,1,3])
    # clf=ExtraTreesClassifier(n_estimators=500, bootstrap=False, criterion='entropy',max_depth=30, min_samples_split=2,min_samples_leaf=2,n_jobs=-1)
    tiempo_inicio=timeit.default_timer()
    clf.fit(X_train, y_train) # realizamos el entrenamiento
    tiempo_entreno=timeit.default_timer()-tiempo_inicio
    tiempo_inicio_=timeit.default_timer()
    y_pred = clf.predict(X_test)
    tiempo_pred=timeit.default_timer()-tiempo_inicio_
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    print('tiempo de entreno: ', tiempo_entreno)
    print('tiempo de prediccion: ', tiempo_pred)
    cm = confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(cm)
    tipo_golpe(y_pred)
    pass

# media 54.38
# best_params() # 20,2,1,entropy
dt_classifier()
# iter_clf()