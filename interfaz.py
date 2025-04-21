import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image, ImageTk

modelo = load_model('modelo_entrenado.keras')
altura, anchura = 224, 224

def clasificar():
    ruta = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.png *.jpeg")])
    if ruta:
        # Procesamiento para predicción
        img = load_img(ruta, target_size=(altura, anchura))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediccion = modelo.predict(img_array)
        clase = np.argmax(prediccion)

        if clase == 0:
            resultado.set("🔥 ¡Incendio detectado!")
            resultado_label.config(fg="red")
        else:
            resultado.set("✅ Todo está en orden")
            resultado_label.config(fg="green")

        # Mostrar imagen en interfaz
        img_pil = Image.open(ruta).resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_pil)
        imagen_label.config(image=img_tk)
        imagen_label.image = img_tk

# Crear ventana
ventana = tk.Tk()
ventana.title("Detector de Incendios")
ventana.geometry("600x400")
ventana.resizable(False, False)

# Fondo personalizado
fondo_img = Image.open("fondo.png").resize((600, 400))
fondo_tk = ImageTk.PhotoImage(fondo_img)
fondo_label = tk.Label(ventana, image=fondo_tk)
fondo_label.place(x=0, y=0, relwidth=1, relheight=1)

# Contenedor para widgets
marco = tk.Frame(ventana, bg="#FFFFFF", bd=0)
marco.place(relx=0.5, rely=0.5, anchor="center", width=500, height=320)

# Título
titulo = tk.Label(marco, text="Detector de Incendios", font=("Helvetica", 18, "bold"), bg="#FFFFFF")
titulo.pack(pady=(20, 10))

# Resultado
resultado = tk.StringVar()
resultado_label = tk.Label(marco, textvariable=resultado, font=("Helvetica", 14), bg="#FFFFFF")
resultado_label.pack(pady=(5, 15))

# Botón
boton = tk.Button(marco, text="Seleccionar Imagen", command=clasificar,
                  bg="#007acc", fg="white", font=("Helvetica", 12), relief="flat", padx=10, pady=5)
boton.pack()

# Imagen miniatura
imagen_label = tk.Label(marco, bg="#000000")
imagen_label.pack(pady=10)

ventana.mainloop()
