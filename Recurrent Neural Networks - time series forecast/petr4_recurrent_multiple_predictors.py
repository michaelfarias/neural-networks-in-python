# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:27:32 2019

@author: Michael
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()
base_train = base.iloc[:, 1:7].values

normalizer = MinMaxScaler(feature_range=(0, 1))
base_train_normalized = normalizer.fit_transform(base_train)

normalizer_predictors = MinMaxScaler(feature_range = (0, 1))
normalizer_predictors.fit_transform(base_train[:, 0:1])

predictors = []
real_price = []

for i in range(90, 1242):
    predictors.append(base_train_normalized[i - 90 : i, 0:6])
    real_price.append(base_train_normalized[i, 0])

predictors, real_price = np.array(predictors), np.array(real_price)

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences=True,
                   input_shape = (predictors.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences=True))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.3))


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation='sigmoid'))

regressor.compile(optimizer='adam', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'weights.h5', monitor = 'loss', save_best_only = True, 
                      verbose = 1)
regressor.fit(predictors, real_price, epochs = 100, batch_size = 32, callbacks = [es, rlr, mcp])                        



base_test = pd.read_csv('petr4_teste.csv')

real_price_test = base_test.iloc[:, 1:2].values
frames = [base, base_test]
complete_base = pd.concat(frames)
complete_base = complete_base.drop('Date', axis = 1)

inputs = complete_base[len(complete_base) - len(base_test) - 90 : ].values
inputs = normalizer.transform(inputs)


x_test = []

for i in range(90, 112):
    x_test.append(inputs[i - 90 : i, 0:6])

x_test = np.array(x_test)

predictors = regressor.predict(x_test)
predictors = normalizer_predictors.inverse_transform(predictors)


predictors.mean()
real_price_test.mean()

plt.plot(real_price_test, color = 'red', label = 'Real price')
plt.plot(predictors, color = 'blue', label = 'Predictions')
plt.title('Stock price forecast')
plt.xlabel('time')
plt.ylabel('Yahoo value')
plt.legend()
plt.show()