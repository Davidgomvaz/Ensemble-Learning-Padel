# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:11:59 2023

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

#%% Definimos el clasificador y mostramos su resultado

from sklearn import model_selection
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
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
    param_grid={'n_neighbors': [1,2,3,4,5,6,7,8,9.10],
                'p': [1,2]}
    model = model_selection.GridSearchCV(estimator= KNeighborsClassifier(),
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=i)
        knn_clf = KNeighborsClassifier(n_neighbors=1,p=1)
        knn_clf.fit(X_train, y_train)
        y_pred = knn_clf.predict(X_test)
        score=accuracy_score(y_test, y_pred)*100
        print(i, score)
        result.append(score)
    print("Precision: %.3f%% (+/-%.3f)" % (mean(result) , std(result)))
    pyplot.figure()
    pyplot.boxplot(result)
    pyplot.title('Resultado KNN')
    pyplot.ylabel('Precisión(%)')
    
def param_results(k,p):
    best_predict=0
    for i in k:
        for j in p:
            knn_clf = KNeighborsClassifier(n_neighbors=i,p=j)
            knn_clf.fit(X_train, y_train)
            y_pred = knn_clf.predict(X_test)
            prediction=accuracy_score(y_test, y_pred)*100
            print(knn_clf.__class__.__name__, prediction, i,j)
            if prediction>best_predict:
                best_predict=prediction
                best_weights=[i,j]
    print("El mejor valor es",best_predict,best_weights)
    pass

# Funcion para definir y entrenar el clasificador 
def knn_classifier():
    knn_clf = KNeighborsClassifier(n_neighbors=1,p=1)
    tiempo_inicio=timeit.default_timer()
    knn_clf.fit(X_train, y_train) # realizamos el entrenamiento
    tiempo_entreno=timeit.default_timer()-tiempo_inicio
    tiempo_inicio_=timeit.default_timer()
    y_pred = knn_clf.predict(X_test)
    tiempo_pred=timeit.default_timer()-tiempo_inicio_
    print(knn_clf.__class__.__name__, accuracy_score(y_test, y_pred))
    print('tiempo de entreno: ', tiempo_entreno)
    print('tiempo de prediccion: ', tiempo_pred)
    cm = confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(cm)
    pass

k=[1,2,3,4,5,6,7,8,9,10]
p=[1,2]
# param_results(k,p)
# best_params()
# knn_classifier()
iter_clf()