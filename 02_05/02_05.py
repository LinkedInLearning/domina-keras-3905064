import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Dropout, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile
import os

url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
urllib.request.urlretrieve(url, 'ml-100k.zip')
with zipfile.ZipFile('ml-100k.zip', 'r') as zip_ref:
    zip_ref.extractall()

datos = pd.read_csv('ml-100k/u.data', sep='\t', names=['usuario_id', 'pelicula_id', 'calificacion', 'timestamp'])

usuarios = datos['usuario_id'].unique()
peliculas = datos['pelicula_id'].unique()
num_usuarios = len(usuarios) + 1
num_peliculas = len(peliculas) + 1

df_train, df_test = train_test_split(datos, test_size=0.2, random_state=42)

entrada_usuario = Input(shape=(1,))
entrada_pelicula = Input(shape=(1,))

embedding_usuario = Embedding(input_dim=num_usuarios, output_dim=100, embeddings_regularizer='l2')(entrada_usuario)
embedding_pelicula = Embedding(input_dim=num_peliculas, output_dim=100, embeddings_regularizer='l2')(entrada_pelicula)

usuario_flat = Flatten()(embedding_usuario)
usuario_flat = Dropout(0.5)(usuario_flat)
pelicula_flat = Flatten()(embedding_pelicula)
pelicula_flat = Dropout(0.5)(pelicula_flat)

concatenado = Concatenate()([usuario_flat, pelicula_flat])
hidden = Dense(128, activation='relu')(concatenado)
hidden = Dense(64, activation='relu')(hidden)
salida = Dense(1, activation='sigmoid')(hidden)
salida = salida * 4 + 1

modelo = Model(inputs=[entrada_usuario, entrada_pelicula], outputs=salida)
modelo.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=[MeanSquaredError()])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

modelo.fit(
    [df_train['usuario_id'], df_train['pelicula_id']],
    df_train['calificacion'],
    epochs=50,
    batch_size=128,
    verbose=1,
    validation_split=0.1,
    callbacks=[early_stopping]
)

prediccion1 = modelo.predict([np.array([1]), np.array([50])])
prediccion2 = modelo.predict([np.array([10]), np.array([20])])
prediccion3 = modelo.predict([np.array([25]), np.array([35])])

prediccion1 = np.clip(prediccion1, 1, 5)
prediccion2 = np.clip(prediccion2, 1, 5)
prediccion3 = np.clip(prediccion3, 1, 5)

print(f'Predicción de calificación para Usuario 1 y Película 50: {prediccion1[0][0]:.2f}')
print(f'Predicción de calificación para Usuario 10 y Película 20: {prediccion2[0][0]:.2f}')
print(f'Predicción de calificación para Usuario 25 y Película 35: {prediccion3[0][0]:.2f}')