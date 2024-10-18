import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
import re

max_palabras = 10000
longitud_maxima = 100
dim_embedding = 128

(X_entrenamiento, y_entrenamiento), (X_prueba, y_prueba) = imdb.load_data(num_words=max_palabras)

X_entrenamiento = pad_sequences(X_entrenamiento, maxlen=longitud_maxima)
X_prueba = pad_sequences(X_prueba, maxlen=longitud_maxima)

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Z0-9\s]", '', texto)
    return texto

def texto_a_secuencia(texto, word_index):
    secuencia = []
    for palabra in texto.split():
        if palabra in word_index and word_index[palabra] < max_palabras:
            secuencia.append(word_index[palabra] + 3)
        else:
            secuencia.append(2)
    return secuencia

modelo = Sequential()
modelo.add(Embedding(max_palabras, dim_embedding, input_length=longitud_maxima))
modelo.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01)))
modelo.add(Dropout(0.6))
modelo.add(Dense(1, activation='sigmoid'))

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

modelo.fit(X_entrenamiento, y_entrenamiento, epochs=10, batch_size=64,
           validation_data=(X_prueba, y_prueba), callbacks=[early_stopping])

puntuacion, precision = modelo.evaluate(X_prueba, y_prueba)
print(f"Precisión en el conjunto de prueba: {precision:.2f}")

word_index = imdb.get_word_index()

nuevos_comentarios = [
    "The movie was fantastic, really enjoyed the storyline and the performances!",
    "This was the worst movie I have ever seen, completely terrible and boring.",
    "An excellent movie with great acting and a compelling plot.",
]

X_nuevos = []
for comentario in nuevos_comentarios:
    comentario_limpio = limpiar_texto(comentario)
    secuencia = texto_a_secuencia(comentario_limpio, word_index)
    X_nuevos.append(secuencia)

X_nuevos = pad_sequences(X_nuevos, maxlen=longitud_maxima)

predicciones = modelo.predict(X_nuevos)

for i, comentario in enumerate(nuevos_comentarios):
    sentimiento = 'Positivo' if predicciones[i][0] > 0.5 else 'Negativo'
    print(f"Comentario: {comentario}")
    print(f"Predicción: {sentimiento} ({predicciones[i][0]:.2f})\n")