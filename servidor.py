import os
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request

from PIL import Image, ImageOps


# ============================================================
# CONFIGURACIÓN
# ============================================================

app = Flask(__name__)

MODEL_PATH = "modelos/emnist_model.keras"

IMAGE_SIZE = (28, 28)


# Mapeo EMNIST Balanced
# Índices oficiales de EMNIST Balanced
EMNIST_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"
]


# ============================================================
# CARGAR MODELO
# ============================================================

print("=" * 60)
print("EMNIST - SERVIDOR DE PREDICCIÓN")
print("=" * 60)

print("\nCargando modelo...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Modelo cargado correctamente.")


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preparar_imagen(archivo):

    # Abrir imagen
    imagen = Image.open(archivo)

    print("\nImagen recibida")
    print("Tamaño original:", imagen.size)
    print("Modo original:", imagen.mode)

    # Convertir a escala de grises
    imagen = imagen.convert("L")

    # Invertir colores
    imagen = ImageOps.invert(imagen)

    # Redimensionar
    imagen = imagen.resize(IMAGE_SIZE)

    # Convertir a NumPy
    imagen_array = np.array(imagen, dtype=np.float32)

    # Normalizar
    imagen_array = imagen_array / 255.0

    # Agregar canal
    imagen_array = np.expand_dims(imagen_array, axis=-1)

    # Agregar dimensión del batch
    imagen_array = np.expand_dims(imagen_array, axis=0)

    print("Forma final:", imagen_array.shape)

    return imagen_array


# ============================================================
# PREDICCIÓN
# ============================================================

def realizar_prediccion(imagen):

    predicciones = model.predict(imagen, verbose=0)

    probabilidades = predicciones[0]

    indice = int(np.argmax(probabilidades))

    confianza = float(probabilidades[indice]) * 100

    if indice < len(EMNIST_CLASSES):
        clase = EMNIST_CLASSES[indice]
    else:
        clase = str(indice)

    return clase, confianza


# ============================================================
# RUTA PRINCIPAL
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# RUTA DE PREDICCIÓN
# ============================================================

@app.route("/predecir", methods=["POST"])
def predecir():

    if "imagen" not in request.files:

        return render_template(
            "index.html",
            error="No se recibió ninguna imagen."
        )

    archivo = request.files["imagen"]

    if archivo.filename == "":

        return render_template(
            "index.html",
            error="No se seleccionó ninguna imagen."
        )

    try:

        # Preparar imagen
        imagen = preparar_imagen(archivo)

        # Predicción
        clase, confianza = realizar_prediccion(imagen)

        return render_template(
            "index.html",
            resultado=clase,
            confianza=f"{confianza:.2f}%"
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Error procesando la imagen: {e}"
        )


# ============================================================
# EJECUTAR SERVIDOR
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SERVIDOR INICIADO")
    print("=" * 60)

    print("\nAbre en tu navegador:")
    print("http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
