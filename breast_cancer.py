# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:32:45 2019

@author: Michael
"""

import numpy as np
from sklearn import datasets

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def sigmoid_derivada(sig):
    return sig * (1 - sig)


base = datasets.load_breast_cancer()
entradas = base.data
valores_saida = base.target
saidas = np.empty([569, 1], dtype = int)

for i in range(569):
    saidas[ i ] = valores_saida[ i ]

pesos0 = 2 * np.random.random((30, 5)) - 1
pesos1 = 2 * np.random.random((5, 1)) - 1

epocas = 10000
taxa_aprendizagem = 0.3
momento = 1


for i in range(epocas):
    camada_entrada = entradas
    soma_sinapse0 = np.dot(camada_entrada, pesos0)
    camada_oculta = sigmoid(soma_sinapse0)
    
    soma_sinapse1 = np.dot(camada_oculta, pesos1)
    camada_saida = sigmoid(soma_sinapse1)
    
    erro_camada_saida = saidas - camada_saida
    media_absoluta = np.mean(np.abs(erro_camada_saida))
    print("Erro: " + str(media_absoluta))
    
    derivada_saida = sigmoid_derivada(camada_saida)
    delta_saida = erro_camada_saida * derivada_saida
    
    pesos1_transposta = pesos1.T
    delta_saida_x_peso = delta_saida.dot(pesos1_transposta)
    delta_camada_oculta = delta_saida_x_peso * sigmoid_derivada(camada_oculta)
    
    camada_oculta_transposta = camada_oculta.T
    pesos_novos1 = camada_oculta_transposta.dot(delta_saida)
    pesos1 = (pesos1 * momento) + (pesos_novos1 * taxa_aprendizagem)
    
    camada_entrada_transposta = camada_entrada.T
    pesos_novos0 = camada_entrada_transposta.dot(delta_camada_oculta)
    pesos0 = (pesos0 * momento) + (pesos_novos0 * taxa_aprendizagem)