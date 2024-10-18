import numpy as np
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import RMSprop

ruta = 'texto.txt'
with open(ruta, 'r') as archivo:
    texto = archivo.read().lower()

tokenizador = RegexpTokenizer(r'\w+')
palabras = tokenizador.tokenize(texto)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(palabras)
total_palabras = len(tokenizer.word_index) + 1

LONGITUD_SECUENCIA = 5
secuencias = []
for i in range(LONGITUD_SECUENCIA, len(palabras)):
    secuencia = palabras[i-LONGITUD_SECUENCIA:i+1]
    secuencias.append(secuencia)

secuencias = tokenizer.texts_to_sequences(secuencias)
secuencias = np.array(secuencias)
X, Y = secuencias[:, :-1], secuencias[:, -1]

X_one_hot = np.zeros((X.shape[0], X.shape[1], total_palabras), dtype=np.bool_)
for i, secuencia in enumerate(X):
    for t, palabra in enumerate(secuencia):
        X_one_hot[i, t, palabra] = 1

Y = np.eye(total_palabras)[Y]

def predecir_siguiente_palabra(modelo, texto, tokenizer, LONGITUD_SECUENCIA, total_palabras):
    tokens = tokenizer.texts_to_sequences([texto.split()[-LONGITUD_SECUENCIA:]])[0]
    tokens_padded = np.pad(tokens, (LONGITUD_SECUENCIA - len(tokens), 0), mode='constant')
    tokens_one_hot = np.zeros((1, LONGITUD_SECUENCIA, total_palabras), dtype=np.bool_)
    for t, palabra in enumerate(tokens_padded):
        tokens_one_hot[0, t, palabra] = 1
    prediccion = modelo.predict(tokens_one_hot)
    palabra_predicha = tokenizer.index_word[np.argmax(prediccion)]
    return palabra_predicha

modelo = Sequential([
    LSTM(128),
    Dense(total_palabras, activation='softmax')
])

modelo.compile(optimizer=RMSprop(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
modelo.summary()

modelo.fit(X_one_hot, Y, batch_size=128, epochs=20, validation_split=0.2)


texto_prueba = "el clima parece"
print(f"Siguiente palabra predicha: {predecir_siguiente_palabra(modelo, texto_prueba, tokenizer, LONGITUD_SECUENCIA, total_palabras)}")