# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:52:46 2023

@author: david
"""

#%% Preparamos los datos de los golpes de padel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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

#%% Definimos los clasificadores que utilizaremos y mostramos sus resultados

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

def base_classifiers():
    mlp = MLPClassifier(hidden_layer_sizes=(1000,500), activation='relu',batch_size=30,max_iter=70, random_state=5)
    svm_clf = svm.SVC(C=5, decision_function_shape='ovr', kernel='rbf', probability=True) #tiene dos posibilidades
    knn_clf = KNeighborsClassifier(n_neighbors=1,p=1)
    dt_clf = DecisionTreeClassifier(max_depth=20,min_samples_split=4,min_samples_leaf=1,criterion='entropy')
    #ert_clf=ExtraTreesClassifier(n_estimators=500,criterion='entropy',max_depth=None, min_samples_split=2,n_jobs=-1)
    # for clf in (svm_clf, dt_clf, knn_clf,ert_clf): # por cada algoritmo
    #     clf.fit(X_train, y_train) # realizamos el entrenamiento
    #     y_pred = clf.predict(X_test) # predicciones con datos de test
    #     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    return mlp,svm_clf,knn_clf,dt_clf


#%% Definimos el ensamble de clasificador y mostramos los resultados
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
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
def best_params(pesos_clf1,pesos_clf2,pesos_clf3,pesos_clf4):
    mlp,svm_clf,knn_clf,dt_clf=base_classifiers()
    best_predict=0
    for i in pesos_clf1:
        for j in pesos_clf2:
            for k in pesos_clf3:
                for l in pesos_clf4:
                    voting_clf_hard = VotingClassifier(
                        estimators=[('mlp', mlp),('svc', svm_clf), ('dt', dt_clf), ('knn', knn_clf)],
                        voting='soft', weights=[i,j,k,l])
                    voting_clf_hard.fit(X_train, y_train)
                    y_pred = voting_clf_hard.predict(X_test)
                    prediction=accuracy_score(y_test, y_pred)
                    print(voting_clf_hard.__class__.__name__, prediction, i, j, k, l)
                    if prediction>best_predict:
                        best_predict=prediction
                        best_weights=[i,j,k,l]
    print("El mejor valor es",best_predict,best_weights)
    pass

# Funcion para encontrar los mejores parametros
# def best_params():
#     mlp,svm_clf,knn_clf,dt_clf=base_classifiers()
#     param_grid={'weights': [[1,1,1,1],[2,2,2,1]]}
#     model = model_selection.GridSearchCV(estimator= VotingClassifier(estimators=[('mlp', mlp),('svc', svm_clf), ('dt', dt_clf), ('knn', knn_clf)],voting='hard'),
#                                           param_grid=param_grid,
#                                           scoring="accuracy",
#                                           cv=5)
#     model.fit(X_train, y_train)
#     print("val. score: %s" % model.best_score_)
#     print("test score: %s" % model.score(X_test, y_test))
#     print("Mejores parámetros:", model.best_params_)
#     pass

def iter_clf():
    result=list()
    mlp,svm_clf,knn_clf,dt_clf=base_classifiers()
    for i in range(5):
        voting_clf_hard = VotingClassifier(
        estimators=[('mlp', mlp),('svc', svm_clf), ('dt', dt_clf), ('knn', knn_clf)],
        voting='soft', weights=[3,3,1,3])
        voting_clf_hard.fit(X_train, y_train)
        y_pred = voting_clf_hard.predict(X_test)
        score=accuracy_score(y_test, y_pred)*100
        print(i, score)
        result.append(score)
    print("Precision: %.3f%% (+/-%.3f)" % (mean(result) , std(result)))
    pyplot.figure()
    pyplot.boxplot(result)
    pyplot.title('Resultado Clasificador por Votación')
    pyplot.ylabel('Precisión(%)')

# Funcion para definir y entrenar el clasificador por votacion hard
def voting_hard():
    mlp,svm_clf,knn_clf,dt_clf=base_classifiers()
    voting_clf_hard = VotingClassifier(
    estimators=[('mlp', mlp),('svc', svm_clf), ('dt', dt_clf), ('knn', knn_clf)],
    voting='hard', weights=[3,3,1,3])
    tiempo_inicio=timeit.default_timer()
    voting_clf_hard.fit(X_train, y_train)
    tiempo_entreno=timeit.default_timer()-tiempo_inicio
    tiempo_inicio_=timeit.default_timer()
    y_pred_hard = voting_clf_hard.predict(X_test)
    tiempo_pred=timeit.default_timer()-tiempo_inicio_
    print(voting_clf_hard.__class__.__name__, accuracy_score(y_test, y_pred_hard))
    print('tiempo de entreno: ', tiempo_entreno)
    print('tiempo de prediccion: ', tiempo_pred)
    cm = confusion_matrix(y_test,y_pred_hard)
    plot_confusion_matrix(cm)
    pass

# Funcion para definir y entrenar el clasificador por votacion soft
def voting_soft():
    mlp,svm_clf,knn_clf,dt_clf=base_classifiers()
    voting_clf_hard = VotingClassifier(
    estimators=[('mlp', mlp),('svc', svm_clf), ('dt', dt_clf), ('knn', knn_clf)],
    voting='soft', weights=[3,3,1,3])
    tiempo_inicio=timeit.default_timer()
    voting_clf_hard.fit(X_train, y_train)
    tiempo_entreno=timeit.default_timer()-tiempo_inicio
    tiempo_inicio_=timeit.default_timer()
    y_pred_hard = voting_clf_hard.predict(X_test)
    tiempo_pred=timeit.default_timer()-tiempo_inicio_
    print(voting_clf_hard.__class__.__name__, accuracy_score(y_test, y_pred_hard))
    print('tiempo de entreno: ', tiempo_entreno)
    print('tiempo de prediccion: ', tiempo_pred)
    cm = confusion_matrix(y_test,y_pred_hard)
    plot_confusion_matrix(cm)
    pass

# Llamamos a las funciones deseadas
pesos_svm=[1, 2]
pesos_dt=[1, 2]
pesos_knn=[1, 2]
pesos_mlp=[1, 2]
# print("\n Hard voting classifier \n")
# best_params(pesos_mlp,pesos_svm,pesos_dt,pesos_knn)
# iter_clf()
print("\n Hard voting classifier \n")
voting_hard()
print("\n Soft voting classifier \n")
voting_soft()

