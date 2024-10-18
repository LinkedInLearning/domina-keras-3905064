import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD

precios = np.array([10, 20, 30, 40, 50], dtype=float)
ingresos = np.array([100, 200, 300, 400, 500], dtype=float)

precios = precios / 50
ingresos = ingresos / 500

modelo = Sequential()
modelo.add(Input(shape=[1]))
modelo.add(Dense(units=1))

optimizador = SGD(learning_rate=0.01)
modelo.compile(optimizer=optimizador, loss='mean_squared_error')

modelo.fit(precios, ingresos, epochs=500)

precio_nuevo = np.array([60], dtype=float)
precio_nuevo = precio_nuevo / 50

ingreso_predicho = modelo.predict(precio_nuevo)

ingreso_predicho = ingreso_predicho * 500

print(f"El ingreso predicho para un precio de 60 es: {ingreso_predicho[0][0]}")