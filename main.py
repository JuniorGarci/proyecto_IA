import cv2
import numpy as np
import tensorflow as tf
import os

# Load saved model from PC
model = tf.keras.models.load_model(r'C:\Users\PC\Documents\pc\ia\model\model.h5')
model.summary()
data_dir = r'C:\Users\PC\Documents\pc\ia\asl_alphabet_train\asl_alphabet_train'

# Obtener los nombres de las carpetas en el directorio
labels = sorted(os.listdir(data_dir))
labels[-1] = 'Nothing'
print(labels)

# Initiating the video source, 0 for internal camera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Voltear la imagen horizontalmente
    frame = cv2.flip(frame, 1)

    # Dibujar un título en la ventana principal
    title = "Detector de Lenguaje de Senyas - Proyecto IA"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 0, 0)  # Azul
    thickness = 2
    # Posición del título
    position = (20, 50)
    cv2.putText(frame, title, position, font, font_scale, color, thickness)

    # Dibujar un rectángulo en el área de interés
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 5)

    # Región de interés
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (50, 50))
    cv2.imshow('roi', roi)

    img = img / 255.0

    # Hacer predicción sobre el frame actual
    prediction = model.predict(img.reshape(1, 50, 50, 3))
    char_index = np.argmax(prediction)
    confidence = round(prediction[0, char_index] * 100, 1)
    predicted_char = labels[char_index]

    # Mostrar el texto solo si la confianza es superior al 80%
    if confidence > 80 and predicted_char != "Nothing":
        msg = predicted_char + ', Conf: ' + str(confidence) + ' %'
        cv2.putText(frame, msg, (80, 80), font, font_scale, (0, 255, 255), thickness)

    # Mostrar el cuadro principal
    cv2.imshow('frame', frame)

    # Cerrar la cámara al presionar 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
