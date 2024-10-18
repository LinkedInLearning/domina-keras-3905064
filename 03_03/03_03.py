import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

dim_entrada = (28, 28, 1)
dim_capa_oculta = 64


entrada = Input(shape=dim_entrada)
plano = Flatten()(entrada)
capa_oculta = Dense(dim_capa_oculta, activation='relu')(plano)
compresion = Dense(dim_capa_oculta, activation='relu')(capa_oculta)
plano_reconstruido = Dense(28*28, activation='sigmoid')(compresion)
salida = Reshape((28, 28, 1))(plano_reconstruido)

autoencoder = Model(entrada, salida)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

imagenes_comprimidas = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(imagenes_comprimidas[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()