import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

datos_ventas = [120, 135, 150, 145, 160, 175, 180, 160, 140, 125, 130, 140] * 6
datos_ventas = np.array(datos_ventas).reshape(-1, 1)

escalador = MinMaxScaler(feature_range=(0, 1))
datos_escalados = escalador.fit_transform(datos_ventas)

def crear_secuencia(datos, secuencia_size):
    X, y = [], []
    for i in range(len(datos) - secuencia_size):
        X.append(datos[i:i+secuencia_size])
        y.append(datos[i+secuencia_size])
    return np.array(X), np.array(y)

secuencia_size = 3
X, y = crear_secuencia(datos_escalados, secuencia_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

modelo = Sequential()
modelo.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
modelo.add(LSTM(100, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(100, return_sequences=False))
modelo.add(Dropout(0.2))
modelo.add(Dense(1))

modelo.compile(optimizer='adam', loss='mean_squared_error')

modelo.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

predicciones = modelo.predict(X_test)
predicciones_invertidas = escalador.inverse_transform(predicciones)

print(predicciones_invertidas)