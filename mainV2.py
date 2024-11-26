import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Label, Button, Canvas
from PIL import Image, ImageTk
import os

# Cargar el modelo
model = tf.keras.models.load_model(r'E:\Downloads\proyectosIA\proyecto_IA\new_model_20ep.h5')
data_dir = r'E:\dataset'
labels = sorted(os.listdir(data_dir))
labels[-1] = 'Nothing'

# Crear la ventana principal
window = Tk()
window.title("Predicción en Tiempo Real")
window.geometry("1000x700")

# Mostrar imagen de referencia
reference_image_path = r'E:\Downloads\proyectosIA\proyecto_IA\american_sign_language.PNG'
ref_img = Image.open(reference_image_path)
ref_img = ref_img.resize((300, 300))  # Ajustar tamaño de la imagen
ref_img_tk = ImageTk.PhotoImage(ref_img)

# Elementos de la interfaz
image_label = Label(window, image=ref_img_tk, text="Guía de Señas", compound="top", font=("Helvetica", 16))
image_label.pack(side="left", padx=10, pady=10)

video_label = Label(window)
video_label.pack(side="right", padx=10, pady=10)

result_label = Label(window, text="Predicción: ", font=("Helvetica", 16))
result_label.pack()

button_start = Button(window, text="Iniciar", command=lambda: start_video())
button_start.pack()

button_stop = Button(window, text="Detener", command=lambda: stop_video())
button_stop.pack()

# Variables globales
cap = None
running = False

def start_video():
    global cap, running
    running = True
    cap = cv2.VideoCapture(0)  # Abrir la cámara
    update_video()

def stop_video():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
    video_label.config(image='')  # Limpiar el video

def predict_frame(frame):
    # Recortar y preprocesar el área de interés
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (50, 50)) / 255.0
    prediction = model.predict(img.reshape(1, 50, 50, 3))
    char_index = np.argmax(prediction)
    confidence = round(prediction[0, char_index] * 100, 1)
    return labels[char_index], confidence

def update_video():
    global cap, running
    if running and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Mostrar el área de interés (ROI) sin efecto espejo
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 2)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            
            # Actualizar el video en la interfaz
            video_label.img_tk = img_tk
            video_label.config(image=img_tk)
            
            # Realizar predicción
            predicted_char, confidence = predict_frame(frame)
            result_label.config(text=f"Predicción: {predicted_char}, Confianza: {confidence}%")
        video_label.after(10, update_video)

# Iniciar la interfaz
window.mainloop()
