# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:41:01 2019

@author: Michael
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def create_network(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer= kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation ,
                            kernel_initializer= kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation='sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    
    return classificador


classificador = KerasClassifier(build_fn= create_network)
parameter = {'batch_size' : [10, 30],
             'epochs': [50, 100],
             'optimizer' : ['adam', 'sgd'], 
             'loss': ['binary_crossentropy', 'hinge'],
             'kernel_initializer': ['random_uniform', 'normal'],
             'activation': ['relu', 'tanh'], 
             'neurons': [16, 8]}



grid_search = GridSearchCV(estimator=classificador, param_grid= parameter,
                           scoring='accuracy', cv = 3)
grid_search = grid_search.fit(previsores, classe)
best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_
