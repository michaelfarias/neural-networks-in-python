# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:27:39 2019

@author: Michael
"""

import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def sigmoid_derivada(sig):
    return sig * (1 - sig)

#a = sigmoid(0.5)
#b = sigmoid_derivada(a)

#a = sigmoid(50)
    
entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
    
saidas = np.array([[0], [1], [1], [0]])

#pesos0 = np.array([[-0.424, -0.740, -0.961],
 #                  [0.358, -0.577, -0.469]])

#pesos1 = np.array([[-0.017], [-0.893], [0.148]])


pesos0 = 2 * np.random.random((2, 3)) - 1
pesos1 = 2 * np.random.random((3, 1)) - 1

epocas = 1000000
taxa_aprendizagem = 0.6
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    