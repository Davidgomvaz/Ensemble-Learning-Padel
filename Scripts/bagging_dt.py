# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:11:44 2023

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

#%% Definimos el clasificador base y mostramos su resultado

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def base_classifier():
    dt_clf = DecisionTreeClassifier(max_depth=20,min_samples_split=4,min_samples_leaf=1,criterion='entropy')
    dt_clf.fit(X_train, y_train) # realizamos el entrenamiento
    y_pred = dt_clf.predict(X_test)
    print(dt_clf.__class__.__name__, accuracy_score(y_test, y_pred))
    return dt_clf

#%% Definimos el ensamble de clasificador y mostramos su resultado

from sklearn.ensemble import BaggingClassifier
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

# Funcion para encontrar los mejores parametros
def best_params():
    dt_clf=base_classifier()
    param_grid={'estimator': [dt_clf],
                'n_estimators': [10,100,250,500],
                'max_features': [25,50,100,200],
                'bootstrap': [False]}
    model = model_selection.GridSearchCV(estimator= BaggingClassifier(estimator=dt_clf),
                                          param_grid=param_grid,
                                          scoring="accuracy",
                                          cv=5)
    model.fit(X_train, y_train)
    print("val. score: %s" % model.best_score_)
    print("test score: %s" % model.score(X_test, y_test))
    print("Mejores parámetros:", model.best_params_)
    pass

def param_results(n_estimators,max_feature):
    best_predict=0
    dt_clf=base_classifier()
    for i in n_estimators:
        for j in max_feature:
            past_clf = BaggingClassifier(estimator=dt_clf, n_estimators=i,max_features=j, bootstrap=False,n_jobs=-1)
            past_clf.fit(X_train, y_train)
            y_pred = past_clf.predict(X_test)
            prediction=accuracy_score(y_test, y_pred)*100
            print(past_clf.__class__.__name__, prediction, i,j)
            if prediction>best_predict:
                best_predict=prediction
                best_weights=[i,j]
    print("El mejor valor es",best_predict,best_weights)
    pass

def iter_clf():
    result=list()
    dt_clf=base_classifier()
    for i in range(10):
        bag_clf = BaggingClassifier(estimator=dt_clf, n_estimators=500, max_features=25, bootstrap=False, n_jobs=-1)
        bag_clf.fit(X_train, y_train)
        y_pred = bag_clf.predict(X_test)
        score=accuracy_score(y_test, y_pred)*100
        print(i, score)
        result.append(score)
    print("Precision: %.3f%% (+/-%.3f)" % (mean(result) , std(result)))
    pyplot.figure()
    pyplot.boxplot(result)
    pyplot.title('Resultado Pasting con Árbol de Decisión')
    pyplot.ylabel('Precisión(%)')

# Funcion para definir y entrenar el clasificador bagging
def bag_classifier():
    dt_clf=base_classifier()
    bag_clf = BaggingClassifier(estimator=dt_clf, n_estimators=500, max_features=25, bootstrap=False, n_jobs=-1)
    tiempo_inicio=timeit.default_timer()
    bag_clf.fit(X_train, y_train)
    tiempo_entreno=timeit.default_timer()-tiempo_inicio
    tiempo_inicio_=timeit.default_timer()
    y_pred = bag_clf.predict(X_test)
    tiempo_pred=timeit.default_timer()-tiempo_inicio_
    print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))
    print('tiempo de entreno: ', tiempo_entreno)
    print('tiempo de prediccion: ', tiempo_pred)
    cm = confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(cm)
    pass

n_estimators=[10,100,250,500]
max_feature=[25]
param_results(n_estimators,max_feature)
# bag_classifier()
# best_params()
# iter_clf()

