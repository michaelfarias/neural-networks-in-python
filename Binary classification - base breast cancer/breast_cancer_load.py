# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:15:42 2019

@author: Michael
"""

import numpy as np
import pandas as pd
from keras.models import model_from_json

file = open('classificador_breast.json', 'r')
struct_network = file.read()
file.close()

classificador = model_from_json(struct_network)
classificador.load_weights('classificador_breast.h5')


new = np.array([[15.80, 8.34, 118, 900, 0.1, 0.26, 0.08, 0.134, 0.178,
                0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                0.03, 0.007, 23.15, 16.64, 178.5, 2019, 0.14, 0.185,
                0.84, 158, 0.363]])

prevision = classificador.predict(new)
prevision = (prevision > 0.5)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')
classificador.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['binary_accuracy'])
resultado = classificador.evaluate(previsores, classe)