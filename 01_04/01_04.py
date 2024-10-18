import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
datos = pd.DataFrame({'texto': newsgroups.data, 'categoria': newsgroups.target})


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    tokens = texto.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    texto = ' '.join(tokens)
    return texto

datos['texto_limpio'] = datos['texto'].apply(limpiar_texto)

tokenizador = Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizador.fit_on_texts(datos['texto_limpio'])
secuencias = tokenizador.texts_to_sequences(datos['texto_limpio'])

entradas = pad_sequences(secuencias, maxlen=300, padding='post', truncating='post')

etiqueta_encoder = LabelEncoder()
etiquetas = etiqueta_encoder.fit_transform(datos['categoria'])
etiquetas = tf.keras.utils.to_categorical(etiquetas, num_classes=20)

entradas_entrenamiento, entradas_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
    entradas, etiquetas, test_size=0.2, random_state=42
)


etiquetas_entrenamiento_clases = np.argmax(etiquetas_entrenamiento, axis=1)
pesos_clase = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(etiquetas_entrenamiento_clases),
    y=etiquetas_entrenamiento_clases
)
pesos_clase_dict = {i: peso for i, peso in enumerate(pesos_clase)}

modelo = Sequential()
modelo.add(Embedding(input_dim=20000, output_dim=128))
modelo.add(Bidirectional(LSTM(128, return_sequences=True)))
modelo.add(Dropout(0.5))
modelo.add(Bidirectional(LSTM(64)))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(20, activation='softmax'))

modelo.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

modelo.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

historial = modelo.fit(
    entradas_entrenamiento, etiquetas_entrenamiento,
    epochs=10,
    batch_size=128,
    validation_data=(entradas_prueba, etiquetas_prueba),
    callbacks=[early_stopping],
    class_weight=pesos_clase_dict
)

perdida, exactitud = modelo.evaluate(entradas_prueba, etiquetas_prueba)
print(f'Exactitud en el conjunto de prueba: {exactitud * 100:.2f}%')