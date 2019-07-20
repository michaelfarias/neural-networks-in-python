# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:49:10 2019

@author: Michael
"""

import pandas as pd

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_train, previsores_test, class_train, class_test = train_test_split(previsores, classe, test_size = 0.25)


import keras
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
classificador.add(Dense(units=16, activation='relu',
                        kernel_initializer='random_uniform', input_dim = 30))
classificador.add(Dense(units=16, activation='relu',
                        kernel_initializer='random_uniform'))
classificador.add(Dense(units=1, activation = 'sigmoid'))

#classificador.compile(optimizer='adam', loss='binary_crossentropy',
#                     metrics = ['binary_accuracy'])

optimize = keras.optimizers.Adam(lr = 0.001,  decay = 0.0001, clipvalue = 0.5)

classificador.compile(optimizer=optimize, loss='binary_crossentropy',
                     metrics = ['binary_accuracy'])



#classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['binary_accuracy'])
classificador.fit(previsores_train, class_train, batch_size=10, epochs=100)



weights0 = classificador.layers[0].get_weights()
print(weights0)
print(len(weights0))
weights1 = classificador.layers[1].get_weights()

previsoes = classificador.predict(previsores_test)
previsoes = (previsoes > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(class_test, previsoes)
matriz = confusion_matrix(class_test, previsoes)

resultado = classificador.evaluate(previsores_test, class_test)