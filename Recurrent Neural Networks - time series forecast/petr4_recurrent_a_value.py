# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 10:52:22 2019

@author: Michael
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()
base_train = base.iloc[:, 1:2].values

normalizer = MinMaxScaler(feature_range=(0, 1))
base_train_normalized = normalizer.fit_transform(base_train)

predictors = []
real_value = []

for i in range(90, 1242):
    predictors.append(base_train_normalized[i - 90 : i, 0])
    real_value.append(base_train_normalized[i, 0])

predictors, real_value = np.array(predictors), np.array(real_value)
predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1))


regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences=True, input_shape = (predictors.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation='linear'))

regressor.compile(optimizer='rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

regressor.fit(predictors, real_value, epochs = 100, batch_size=32)

base_test = pd.read_csv('petr4_teste.csv')

real_price_test = base_test.iloc[:, 1:2].values
complete_base = pd.concat((base['Open'], base_test['Open']), axis = 0)
inputs = complete_base[len(complete_base) - len(base_test) - 90 : ].values
inputs = inputs.reshape(-1, 1)
inputs = normalizer.transform(inputs)


x_test = []

for i in range(90, 112):
    x_test.append(inputs[i - 90 : i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictors = regressor.predict(x_test)
predictors = normalizer.inverse_transform(predictors)

predictors.mean()
real_price_test.mean()

plt.plot(real_price_test, color = 'red', label = 'Real price')
plt.plot(predictors, color = 'blue', label = 'Predictions')
plt.title('Stock price forecast')
plt.xlabel('time')
plt.ylabel('Yahoo value')
plt.legend()
plt.show()




















