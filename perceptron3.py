# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:36:03 2019

@author: Michael
"""


import numpy as np

#AND
#entradas = np.array([[0,0],[0,1],[1,0], [1,1]])
#saidas = np.array([0, 0, 0, 1])
#OR
entradas = np.array([[0,0],[0,1],[1,0], [1,1]])
saidas = np.array([0, 1, 1, 1])
#XOR
#entradas = np.array([[0,0],[0,1],[1,0], [1,1]])
#saidas = np.array([0, 1, 1, 0])

pesos = np.array([0.0, 0.0])
taxa_aprendizagem = 0.1


def step_function(soma):
    if(soma >= 1):
        return 1
    return 0

def calcula_saida(registro):
    s = registro.dot(pesos)
    return step_function(s)


def treinar():
    erro_total =1
    while (erro_total != 0):
        erro_total = 0
        for i in range(len(saidas)):
            saida_calculada =  calcula_saida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saida_calculada)
            erro_total += erro
            for j in range(len(pesos)):
                pesos[j]  =pesos[j] + (taxa_aprendizagem * entradas[i][j]*erro)
                print('Peso atualizado:' + str(pesos[j]))        
                
        print("Total de erros:" + str(erro_total))
        



treinar()
print('Rede neural treinada')
print(calcula_saida(entradas[0]))
print(calcula_saida(entradas[1]))
print(calcula_saida(entradas[2]))
print(calcula_saida(entradas[3]))
