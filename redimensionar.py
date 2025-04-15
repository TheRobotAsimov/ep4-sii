from PIL import Image
import os

def reducir_imagenes_en_directorio(directorio, max_lado=1000):
    for carpeta, _, archivos in os.walk(directorio):
        for archivo in archivos:
            if archivo.endswith((".jpg", ".jpeg", ".png")):
                ruta = os.path.join(carpeta, archivo)
                try:
                    img = Image.open(ruta)
                    # Redimensionar si el ancho o alto supera max_lado
                    if img.width > max_lado or img.height > max_lado:
                        print(f"Redimensionando: {ruta}")
                        img.thumbnail((max_lado, max_lado))
                        img.save(ruta)
                except Exception as e:
                    print(f"Error con {ruta}: {e}")

# Llama la función con tu carpeta de imágenes
reducir_imagenes_en_directorio('D:/bosque/valid')
