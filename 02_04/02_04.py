import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

datos_ingles = ["hello", "how are you", "thank you", "good morning", "good night"]
datos_espanol = ["hola", "cómo estás", "gracias", "buenos días", "buenas noches"]

tokenizador_ingles = Tokenizer()
tokenizador_espanol = Tokenizer()

tokenizador_ingles.fit_on_texts(datos_ingles)
tokenizador_espanol.fit_on_texts(datos_espanol)

vocabulario_ingles = len(tokenizador_ingles.word_index) + 1
vocabulario_espanol = len(tokenizador_espanol.word_index) + 1

secuencias_ingles = tokenizador_ingles.texts_to_sequences(datos_ingles)
secuencias_espanol = tokenizador_espanol.texts_to_sequences(datos_espanol)

longitud_maxima = max(
    max(len(seq) for seq in secuencias_ingles),
    max(len(seq) for seq in secuencias_espanol)
)
secuencias_ingles = pad_sequences(secuencias_ingles, maxlen=longitud_maxima, padding='post')
secuencias_espanol = pad_sequences(secuencias_espanol, maxlen=longitud_maxima, padding='post')

assert secuencias_ingles.shape[0] == secuencias_espanol.shape[0]

entrada = Input(shape=(longitud_maxima,))
embeddings = Embedding(input_dim=vocabulario_ingles, output_dim=64)(entrada)
lstm = LSTM(64, return_sequences=True)(embeddings)
salida = TimeDistributed(Dense(vocabulario_espanol, activation='softmax'))(lstm)

def traducir_frase(frase_ingles):
    secuencia_ingles = tokenizador_ingles.texts_to_sequences([frase_ingles])
    secuencia_ingles = pad_sequences(secuencia_ingles, maxlen=longitud_maxima, padding='post')
    prediccion = modelo.predict(secuencia_ingles)
    palabra_predicha = np.argmax(prediccion, axis=-1)
    traduccion = ' '.join([tokenizador_espanol.index_word.get(idx, '') for idx in palabra_predicha[0] if idx != 0])
    return traduccion.strip()

modelo = Model(inputs=entrada, outputs=salida)
optimizador = Adam(learning_rate=0.001)
modelo.compile(optimizer=optimizador, loss='sparse_categorical_crossentropy')

salidas_espanol = np.expand_dims(secuencias_espanol, -1)

modelo.fit(secuencias_ingles, salidas_espanol, epochs=1000)

frase_a_traducir = "hello"
traduccion = traducir_frase(frase_a_traducir)
print(f"Traducción de '{frase_a_traducir}' es: {traduccion}")