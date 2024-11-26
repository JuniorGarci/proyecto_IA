import cv2
import numpy as np
import tensorflow as tf
import os

# Load saved model from PC
model = tf.keras.models.load_model(r'E:\Downloads\proyectosIA\proyecto_IA\model.h5')
model.summary()
data_dir = r'E:\dataset'

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
    
    # Dibujar un rectángulo en el área de interés
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 5) 
    # Region of interest
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (50, 50))
    cv2.imshow('roi', roi)
    
    
    img = img / 255.0

    # Hacer predicción sobre el frame actual
    
    #La imagen img se redimensiona para ajustarse al formato que espera el modelo. 50x50 píxeles con 3 canales de color
    prediction = model.predict(img.reshape(1, 50, 50, 3))
    #Encuentra el índice del valor más alto en el array de predicciones. Si el array de predicciones es [0.1, 0.05, 0.8, 0.02, 0.03], np.argmax devolvería 2, porque 0.8 es el valor más alto.
    char_index = np.argmax(prediction)
    #Convierte el valor de probabilidad en un porcentaje. En este caso, 0.8 se convierte en 80.0
    confidence = round(prediction[0, char_index] * 100, 1)
    
    #Usa el índice de la clase (char_index) para buscar el nombre de la clase en la lista labels. si labels = ['A', 'B', 'C', 'D', 'E'] y char_index es 2, entonces predicted_char será 'C'
    predicted_char = labels[char_index]

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (0, 255, 255)
    thickness = 2

    # Mostrar el texto solo si la confianza es superior al 80%
    if confidence > 80 and predicted_char != "Nothing":
        msg = predicted_char + ', Conf: ' + str(confidence) + ' %'
        cv2.putText(frame, msg, (80, 80), font, fontScale, color, thickness)
    
    cv2.imshow('frame', frame)
    
    # Cerrar la cámara al presionar 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
