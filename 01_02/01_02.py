# -*- coding: utf-8 -*-
"""01_02.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zlJbHqlCanlMrRAyjhmE01ILDxFS-9rz
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

entradas = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)
salidas = np.array([[0], [0], [0], [0], [1], [1], [1], [1], [1], [1]], dtype=float)

modelo = Sequential()
modelo.add(Input(shape=[1]))
modelo.add(Dense(10, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(1, activation='sigmoid'))

modelo.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

modelo.fit(entradas, salidas, epochs=500, validation_split=0.2)

nueva_solicitud = np.array([11], dtype=float)
resultado_predicho = modelo.predict(nueva_solicitud)

print(f"La predicción para la solicitud es: {'Queja' if resultado_predicho > 0.5 else 'Consulta general'}")