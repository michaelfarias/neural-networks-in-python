# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:43:04 2019

@author: Michael
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu',
                            kernel_initializer= 'normal', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu' ,
                            kernel_initializer= 'normal', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation='sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

new = np.array([[15.80, 8.34, 118, 900, 0.1, 0.26, 0.08, 0.134, 0.178,
                0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                0.03, 0.007, 23.15, 16.64, 178.5, 2019, 0.14, 0.185,
                0.84, 158, 0.363]])


classificador_json = classificador.to_json()

with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
    
classificador.save_weights('classificador_breast.h5')