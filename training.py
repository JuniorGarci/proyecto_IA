import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
import cv2
import pydot


gpu = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[0], True)

#función para cargar los datos y cambiar el tamaño de las imágenes a 50x50
def load_dataset(directory):
  images = []
  labels = []
  for idx, label in enumerate(uniq_labels):
    for file in os.listdir(directory + '/'+label):
      filepath = directory +'/'+ label + "/" + file
      img = cv2.resize(cv2.imread(filepath),(50,50))
      #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
      images.append(img)
      labels.append(idx)
  images = np.asarray(images)
  labels = np.asarray(labels)
  return images, labels

#función para mostrar ejemplos
def display_images(x_data,y_data, title, display_label = True):
    x, y = x_data,y_data
    fig, axes = plt.subplots(5, 8, figsize = (18, 5))
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    fig.suptitle(title, fontsize = 18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB))
        if display_label:
            ax.set_xlabel(uniq_labels[y[i]])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    
#Carga del conjunto de datos en: X_pre and Y_pre
data_dir = r'C:\Users\PC\Documents\pc\ia\asl_alphabet_train\asl_alphabet_train'
uniq_labels = sorted(os.listdir(data_dir))
X_pre, Y_pre = load_dataset(data_dir)
print(X_pre.shape, Y_pre.shape)


#Dividir el conjunto de datos en un 80 % de datos de entrenamiento, un 10 % de validación y un 10 % de datos de prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_pre, Y_pre, test_size = 0.8)
X_test, X_eval, Y_test, Y_eval = train_test_split(X_test, Y_test, test_size = 0.5)

#%matplotlib notebook

#Imprima formas y muestre ejemplos para cada conjunto
"""print("Train images shape",X_train.shape, Y_train.shape)
print("Test images shape",X_test.shape, Y_test.shape)
print("Evaluate image shape",X_eval.shape, Y_eval.shape)
print("Printing the labels",uniq_labels, len(uniq_labels))
display_images(X_train,Y_train,'Samples from Train Set')
display_images(X_test,Y_test,'Samples from Test Set')
display_images(X_eval,Y_eval,'Samples from Validation Set')"""

# Conversión de Y_tes y Y_train en vectores "One hot" usando to_categorical
# Ejemplo de "one hot" => '1' is represented as [0. 1. 0. . . . . 0.]
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_eval = to_categorical(Y_eval)
X_train = X_train / 255.
X_test = X_test/ 255.
X_eval = X_eval/ 255.

# Construyendo el modelo
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation ='relu', input_shape=(50,50,3)),
        tf.keras.layers.Conv2D(16, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(16, (3,3), activation ='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(32, (3,3), activation ='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(25, activation='softmax')
])

model.summary()


#compilación del modelo
#tamaño de lote predeterminado 32
#la tasa de aprendizaje predeterminada es 0,001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#Comienzo del entrenamiento
history = model.fit(X_train, Y_train, epochs=15, verbose=1,
                validation_data=(X_eval, Y_eval))

#Pruebas
model.evaluate(X_test, Y_test)

#Guardar el modelo
model.save(r'C:\Users\PC\Documents\pc\ia\model\model.h5')

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

#Trazando la pérdida de entrenamiento y validación vs. épocas

epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, label = "training loss")
plt.plot(epochs, val_loss, label = "validation  loss")
plt.legend()
plt.show()

#Trazar la precisión del entrenamiento y la validación frente a las épocas
plt.plot(epochs, train_acc, label = "training accuracy")
plt.plot(epochs, val_accuracy, label = "validation  accuracy")
plt.legend()
plt.show()