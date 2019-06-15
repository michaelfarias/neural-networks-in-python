# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:47:14 2019

@author: Michael
"""

import numpy as np

entradas = np.array([-1, 7, 5])
pesos =np.array([0.8, 0.1, 0.0])

def soma(e, p):
    return e.dot(p)
        
s = soma(entradas, pesos)

def step_function(soma):
    if(soma >= 1):
        return 1
    return 0

r = step_function(s)