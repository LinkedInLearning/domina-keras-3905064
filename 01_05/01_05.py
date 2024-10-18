import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

ruta_zip = tf.keras.utils.get_file(
    'cats_and_dogs_filtered.zip',
    'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
    extract=True
)
directorio_base = os.path.join(os.path.dirname(ruta_zip), 'cats_and_dogs_filtered')

directorio_entrenamiento = os.path.join(directorio_base, 'train')
directorio_validacion = os.path.join(directorio_base, 'validation')

tamano_imagen = (150, 150)
batch_tamano = 32
epocas = 50

generador_entrenamiento = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

generador_validacion = ImageDataGenerator(rescale=1./255)

entrenamiento = generador_entrenamiento.flow_from_directory(
    directorio_entrenamiento,
    target_size=tamano_imagen,
    batch_size=batch_tamano,
    class_mode='binary'
)

validacion = generador_validacion.flow_from_directory(
    directorio_validacion,
    target_size=tamano_imagen,
    batch_size=batch_tamano,
    class_mode='binary'
)

modelo = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),  # Incremento de filtros
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),  # Más filtros para mejorar la capacidad
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
modelo.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

historial = modelo.fit(
    entrenamiento,
    epochs=epocas,
    validation_data=validacion,
    callbacks=[early_stopping]
)

plt.plot(historial.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

print(f"Precisión de entrenamiento: {historial.history['accuracy'][-1]*100:.2f}%")
print(f"Precisión de validación: {historial.history['val_accuracy'][-1]*100:.2f}%")