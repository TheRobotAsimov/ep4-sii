#Mi primera red neuronal convolucional
#Importar las librerias necesarias

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers import Convolution2D,MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math

#Ruta de las imágenes

entrenamiento = '/home/jorge/Descargas/sistemas/ep4-sii/bosque/train'
validacion = '/home/jorge/Descargas/sistemas/ep4-sii/bosque/valid'



#Definir los hiperparámetros
epocas = 100
altura,anchura = 224, 224
batch_size = 8
pasos = 100
#Definir la profundidad de la red neuronal convolucional
kernels1 = 16
kernels2 = 32
kernel1_size = (3,3)
kernel2_size = (3,3)
size_pooling = (2,2)
clases = 2


#Generar datos sintéticos para el entrenamiento

entrenar = ImageDataGenerator(rescale=1/255,
                              shear_range=0.20,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              vertical_flip=True)
validar = ImageDataGenerator(rescale=1/255)

imagenes_entrenamiento = entrenar.flow_from_directory(entrenamiento,target_size=(altura,anchura),batch_size=batch_size,class_mode="categorical")
imagenes_validacion=validar.flow_from_directory(validacion,target_size=(altura,anchura),batch_size=batch_size,class_mode="categorical")


steps_per_epoch = math.ceil(imagenes_entrenamiento.samples / batch_size)
validation_steps = math.ceil(imagenes_validacion.samples / batch_size)

#Definir la arquitectura de la red neuronal convolucional
CNN=Sequential()

#Definir la primera capa convolucional
CNN.add(Convolution2D(kernels1,kernel1_size,padding="same",input_shape=(altura,anchura,3),activation="relu"))
CNN.add(MaxPooling2D(pool_size=size_pooling))

#Definir la segunda capa convolucional
CNN.add(Convolution2D(kernels2, kernel2_size, padding="same",activation="relu"))  # OK sin input_shape
CNN.add(MaxPooling2D(pool_size=size_pooling))

#Aplicar flatten
CNN.add(Flatten())

#Conectar con un perceptron multicapa (MLP)
#Primera capa oculta
CNN.add(Dense(128,activation="relu"))
#Apagar un % de neuronas
CNN.add(Dropout(0.5))
#Segunda capa oculta
CNN.add(Dense(64,activation="relu"))
#Apagar un % de neuronas
CNN.add(Dropout(0.3))


#Capa de salida
CNN.add(Dense(2,activation="softmax"))

#Establecer los parámetros iniciales del entrenamiento
CNN.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc","mse"])

#Entrenar la red neuronal convolucional
# Definir callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
check = ModelCheckpoint("mejor_modelo.keras", save_best_only=True)

# Entrenar con callbacks
historico = CNN.fit(
    imagenes_entrenamiento,
    validation_data=imagenes_validacion,
    epochs=epocas,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stop, check],
    verbose=1
)

# Guardar el modelo entrenado
CNN.save('modelo_entrenado.keras')

