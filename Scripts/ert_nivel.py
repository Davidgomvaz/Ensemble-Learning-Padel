# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:31:05 2023

@author: david
"""

#%% Preparamos los datos de los golpes de padel

from sklearn.model_selection import train_test_split
import pandas as pd

# Obtenemos los datos del archivo
datos = pd.read_csv("/Users/david/Desktop/ETSI/4ºCurso/TFg/Golpes/Dataset12.csv")

# Eliminamos las columnas que no nos interesan
datos.drop(columns = ["mano", "reves", "altura", "edad", "sexo", "tipo_golpe","id", "numero_golpe", "tiempo_golpe"], inplace=True)
X = datos.drop(columns = ["nivel"])
y = datos["nivel"]

# Dividimos los datos en datos de entramiento (70% de los datos totales) y datos de prueba 30% 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=5)
print("La forma de los datos es:")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%% Definimos el clasificador base y mostramos su resultado

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def base_classifier():
    dt_clf = DecisionTreeClassifier(max_depth=30, min_samples_leaf=2, min_samples_split=2, criterion="entropy")
    dt_clf.fit(X_train, y_train) # realizamos el entrenamiento
    y_pred = dt_clf.predict(X_test)
    print(dt_clf.__class__.__name__, accuracy_score(y_test, y_pred))
    return dt_clf

#%% Definimos el ensamble de clasificador y mostramos su resultado

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
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

param_grid={'n_estimators': [10,50,100,250,500], #10,50,100,250,500
            'max_samples': [None,1000,750], #100,250,500,
            'max_depth': [30],
            'criterion': ['entropy'],
            'min_samples_split': [2],
            'min_samples_leaf': [2],
            'bootstrap': [True,False]}

def iter_clf():
    result=list()
    for i in range(10):
        rf_clf=ExtraTreesClassifier(n_estimators=500, bootstrap=False, criterion='entropy',max_depth=30, min_samples_split=2,min_samples_leaf=2,n_jobs=-1)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)
        score=accuracy_score(y_test, y_pred)*100
        print(i, score)
        result.append(score)
    print("Precision: %.3f%% (+/-%.3f)" % (mean(result) , std(result)))
    pyplot.figure()
    pyplot.boxplot(result)
    pyplot.title('Resultado Extremely Randomized Trees')
    pyplot.ylabel('Precisión(%)')

def best_params_ert():
    #dt_clf=base_classifier()
    model = model_selection.GridSearchCV(estimator= ExtraTreesClassifier(),
                                          param_grid=param_grid,
                                          scoring="accuracy",
                                          cv=5)
    model.fit(X_train, y_train)
    print("val. score: %s" % model.best_score_)
    print("test score: %s" % model.score(X_test, y_test))
    print("Mejores parámetros:", model.best_params_)
    pass

def param_results(n_estimators):
    best_predict=0
    for i in n_estimators:
        rf_clf = RandomForestClassifier(n_estimators=i, bootstrap=False, criterion='entropy',max_depth=30, min_samples_split=2,min_samples_leaf=2,n_jobs=-1)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)
        prediction=accuracy_score(y_test, y_pred)*100
        print(rf_clf.__class__.__name__, prediction, i)
        if prediction>best_predict:
            best_predict=prediction
            best_weights=[i]
    print("El mejor valor es",best_predict,best_weights)
    pass

# Funcion para definir y entrenar el clasificador bagging
def ert_classifiers():
    dt_clf=base_classifier()
    ert_clf=ExtraTreesClassifier(n_estimators=500, bootstrap=False, criterion='entropy',max_depth=30, min_samples_split=2,min_samples_leaf=2,n_jobs=-1)
    tiempo_inicio=timeit.default_timer()
    ert_clf.fit(X_train, y_train)
    tiempo_entreno=timeit.default_timer()-tiempo_inicio
    tiempo_inicio_=timeit.default_timer()
    y_pred_ert = ert_clf.predict(X_test)
    tiempo_pred=timeit.default_timer()-tiempo_inicio_
    print(ert_clf.__class__.__name__, accuracy_score(y_test, y_pred_ert))
    print(ert_clf.__class__.__name__, roc_auc_score(y_test, y_pred_ert,multi_class='ovr'))
    print('tiempo de entreno: ', tiempo_entreno)
    print('tiempo de prediccion: ', tiempo_pred)
    cm_ert = confusion_matrix(y_test,y_pred_ert)
    plot_confusion_matrix(cm_ert)
    pass


# max_samples=[100,250,500,750,1000]
# n_estimators=[10,50,100,250,500]
# param_results(n_estimators)
# best_params_rf()
ert_classifiers()
# best_params_ert()
# iter_clf()