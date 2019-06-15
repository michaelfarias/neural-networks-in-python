# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:49:40 2019

@author: Michael
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()

camada_entrada = LinearLayer(2)
camada_oculta = SigmoidLayer(3)
camada_saida = SigmoidLayer(1)
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(camada_entrada)
rede.addModule(camada_oculta)
rede.addModule(camada_saida)
rede.addModule(bias1)
rede.addModule(bias2)

entrada_oculta = FullConnection(camada_entrada, camada_oculta)
oculta_saida = FullConnection(camada_oculta, camada_saida)
bias_oculta = FullConnection(bias1, camada_oculta)
bias_saida = FullConnection(bias2, camada_saida)

rede.sortModules()

print(rede)
print(entrada_oculta.params)
print(oculta_saida.params)
print(bias_oculta.params)
print(bias_saida.params)