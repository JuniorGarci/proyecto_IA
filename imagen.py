import cv2
import numpy as np
import tensorflow as tf
import os

# Cargar el modelo
model = tf.keras.models.load_model(r'E:\Downloads\proyectosIA\proyecto_IA\model.h5')
model.summary()


labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Nothing']
print(labels)

# Función para predecir una letra a partir de una imagen
def predict_image(image_path):
    # Cargar la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        return

    # Recortar o redimensionar si es necesario
    img = cv2.resize(img, (50, 50))  # Redimensionar a 50x50
    img = img / 255.0  # Normalizar

    # Hacer predicción
    prediction = model.predict(img.reshape(1, 50, 50, 3)) # Redimensionar para el modelo
    char_index = np.argmax(prediction)
    confidence = round(prediction[0, char_index] * 100, 1)
    predicted_char = labels[char_index]

    # Mostrar resultados
    print(f"Predicción: {predicted_char}, Confianza: {confidence}%")
    if confidence > 80 and predicted_char != "Nothing":
        return predicted_char, confidence
    else:
        return "Predicción poco confiable o nada detectado", confidence

# Ruta de la imagen a analizar
image_path = r'E:\Downloads\letraA.JPG'

# Llamar a la función de predicción
predicted_char, confidence = predict_image(image_path)
print(f"Resultado final: {predicted_char} con confianza del {confidence}%")
