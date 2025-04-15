import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

modelo = load_model('modelo_entrenado.h5')
altura, anchura = 224, 224

def clasificar():
    ruta = filedialog.askopenfilename()
    if ruta:
        img = load_img(ruta, target_size=(altura, anchura))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediccion = modelo.predict(img_array)
        clase = np.argmax(prediccion)

        if clase == 0:
            resultado.set("🔥 Incendio Detectado")
        else:
            resultado.set("✅ No hay incendio")

# Interfaz
ventana = tk.Tk()
ventana.title("Detector de Incendios")

resultado = tk.StringVar()
tk.Button(ventana, text="Seleccionar Imagen", command=clasificar).pack(pady=10)
tk.Label(ventana, textvariable=resultado, font=("Arial", 14)).pack(pady=20)

ventana.mainloop()
