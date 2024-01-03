# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:55:34 2023

@author: david
"""

#%% Preparamos los datos de los golpes de padel
from sklearn.model_selection import train_test_split
import pandas as pd

# Obtenemos los datos del archivo
datos = pd.read_csv("/Users/david/Desktop/ETSI/4ºCurso/TFg/Golpes/Dataset12.csv")

# Eliminamos las columnas que no nos interesan
datos.drop(columns = ["mano", "reves", "altura", "edad", "sexo", "nivel","id", "numero_golpe", "tiempo_golpe"], inplace=True)
X = datos.drop(columns = ["tipo_golpe"])
y = datos["tipo_golpe"]

# Dividimos los datos en datos de entramiento (70% de los datos totales) y datos de prueba 30% 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=5)
print("La forma de los datos es:")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%% definimos los clasificadores

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def base_classifier():
    dt_clf = DecisionTreeClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=4, criterion="entropy")
    dt_clf.fit(X_train, y_train) # realizamos el entrenamiento
    y_pred = dt_clf.predict(X_test) # predicciones con datos de test
    print(dt_clf.__class__.__name__, accuracy_score(y_test, y_pred))
    return dt_clf

#%% Definimos el ensamble de clasificador y mostramos los resultados

from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from numpy import mean
from numpy import std
import timeit

# Funcion para mostrar la matriz de confusion
golpes = ['D','R','DP','RP','GD','GR','GDP','GRP','VD','VR','B','RM','S']
def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
    plt.xticks(range(13), golpes)
    plt.yticks(range(13), golpes)
    pass
    
param_grid={'n_estimators': [50,100,250,500],
            'learning_rate': [0.01,0.05,0.1,0.5,1]}

# Funcion para encontrar los mejores parametros
def best_params():
    dt_clf=base_classifier()
    model = model_selection.GridSearchCV(estimator= AdaBoostClassifier(estimator=dt_clf),
                                         param_grid=param_grid,
                                         scoring="accuracy",
                                         cv=5)
    model.fit(X_train, y_train)
    print("val. score: %s" % model.best_score_)
    print("test score: %s" % model.score(X_test, y_test))
    print("Mejores parámetros:", model.best_params_)
    pass

def iter_clf():
    dt_clf=base_classifier()
    result=list()
    for i in range(5):
        ada_clf = AdaBoostClassifier(estimator=dt_clf,n_estimators=500,learning_rate=1)
        ada_clf.fit(X_train, y_train)
        y_pred = ada_clf.predict(X_test)
        score=accuracy_score(y_test, y_pred)*100
        print(i, score)
        result.append(score)
    print("Precision: %.3f%% (+/-%.3f)" % (mean(result) , std(result)))
    pyplot.figure()
    pyplot.boxplot(result)
    pyplot.title('Resultado AdaBoost')
    pyplot.ylabel('Precisión(%)')

# Funcion para definir y entrenar el clasificador Adaboost
def adaboost_class():
    dt_clf=base_classifier()
    ada_clf = AdaBoostClassifier(estimator=dt_clf,n_estimators=500,learning_rate=1)
    tiempo_inicio=timeit.default_timer()
    ada_clf.fit(X_train, y_train)
    tiempo_entreno=tiempo_inicio-timeit.default_timer()
    tiempo_inicio_=timeit.default_timer()
    y_pred = ada_clf.predict(X_test)
    tiempo_pred=tiempo_inicio_-timeit.default_timer()
    print(ada_clf.__class__.__name__, accuracy_score(y_test, y_pred))
    print('tiempo de entreno: ', tiempo_entreno)
    print('tiempo de prediccion: ', tiempo_pred)
    cm = confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(cm)
    pass

# iter_clf()
adaboost_class()
# best_params()
