import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = "modelos/emnist_model.keras"
RESULTS_DIR = "resultados"

IMAGE_SIZE = (28, 28)


# ============================================================
# CLASES EMNIST BALANCED
# ============================================================

EMNIST_CLASSES = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",

    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z",

    "a", "b", "d", "e", "f", "g", "h",
    "n", "q", "r", "t"
]


# ============================================================
# CARGAR MODELO
# ============================================================

def cargar_modelo():

    print("Cargando modelo...")

    modelo = tf.keras.models.load_model(MODEL_PATH)

    print("Modelo cargado correctamente.")

    return modelo


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preparar_imagen(ruta):

    print()
    print("Imagen original:", ruta)

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe la imagen: {ruta}"
        )

    imagen = Image.open(ruta)

    print("Tamaño original:", imagen.size)
    print("Modo original:", imagen.mode)

    # --------------------------------------------------------
    # Convertir a escala de grises
    # --------------------------------------------------------

    imagen = imagen.convert("L")

    imagen_array = np.array(
        imagen
    ).astype(np.float32)

    print(
        "Pixel mínimo original:",
        f"{imagen_array.min():.2f}"
    )

    print(
        "Pixel máximo original:",
        f"{imagen_array.max():.2f}"
    )

    # --------------------------------------------------------
    # Redimensionar
    # --------------------------------------------------------

    imagen = imagen.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    imagen_array = np.array(
        imagen
    ).astype(np.float32)

    # --------------------------------------------------------
    # Normalizar
    # --------------------------------------------------------

    imagen_array = imagen_array / 255.0

    # --------------------------------------------------------
    # Invertir colores
    # --------------------------------------------------------

    print("Invirtiendo colores...")

    imagen_array = 1.0 - imagen_array

    print(
        "Pixel mínimo normalizado:",
        f"{imagen_array.min():.4f}"
    )

    print(
        "Pixel máximo normalizado:",
        f"{imagen_array.max():.4f}"
    )

    # --------------------------------------------------------
    # Agregar canal
    # --------------------------------------------------------

    imagen_array = np.expand_dims(
        imagen_array,
        axis=-1
    )

    # --------------------------------------------------------
    # Agregar batch
    # --------------------------------------------------------

    imagen_array = np.expand_dims(
        imagen_array,
        axis=0
    )

    print(
        "Forma final para el modelo:",
        imagen_array.shape
    )

    # --------------------------------------------------------
    # Guardar imagen procesada
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    nombre = os.path.splitext(
        os.path.basename(ruta)
    )[0]

    ruta_procesada = os.path.join(
        RESULTS_DIR,
        nombre + "_procesada.png"
    )

    imagen_guardar = (
        imagen_array[0, :, :, 0] * 255
    ).astype(np.uint8)

    Image.fromarray(
        imagen_guardar
    ).save(ruta_procesada)

    print()
    print("Imagen procesada guardada en:")
    print(ruta_procesada)

    return imagen_array, ruta_procesada


# ============================================================
# PREDICCIÓN
# ============================================================

def realizar_prediccion(modelo, imagen):

    probabilidades = modelo.predict(
        imagen,
        verbose=0
    )[0]

    indice = int(
        np.argmax(probabilidades)
    )

    confianza = float(
        probabilidades[indice]
    )

    if indice < len(EMNIST_CLASSES):

        clase = EMNIST_CLASSES[indice]

    else:

        clase = f"Clase {indice}"

    return (
        indice,
        clase,
        confianza,
        probabilidades
    )


# ============================================================
# MOSTRAR RESULTADO
# ============================================================

def mostrar_resultado(
    ruta_original,
    imagen,
    indice,
    clase,
    confianza,
    ruta_procesada,
    etiqueta_real
):

    print()
    print("=" * 60)
    print("RESULTADO DE LA PREDICCIÓN")
    print("=" * 60)

    print(
        "Imagen:",
        ruta_original
    )

    print(
        "Etiqueta real:",
        etiqueta_real
    )

    print(
        "Etiqueta predicha:",
        clase
    )

    print(
        "Índice de clase:",
        indice
    )

    print(
        "Probabilidad:",
        f"{confianza * 100:.2f}%"
    )

    if etiqueta_real.lower() == clase.lower():

        print()
        print("Resultado: CORRECTO")

    else:

        print()
        print("Resultado: INCORRECTO")

    print()
    print(
        "Imagen procesada:",
        ruta_procesada
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Mostrar imagen
    # --------------------------------------------------------

    plt.figure(figsize=(4, 4))

    plt.imshow(
        imagen[0, :, :, 0],
        cmap="gray"
    )

    plt.title(
        f"Real: {etiqueta_real} | "
        f"Predicción: {clase}\n"
        f"Confianza: {confianza * 100:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# TOP 5
# ============================================================

def mostrar_top_predicciones(probabilidades):

    indices = np.argsort(
        probabilidades
    )[::-1][:5]

    print()
    print("Top 5 predicciones:")
    print("-" * 40)

    for posicion, indice in enumerate(
        indices,
        start=1
    ):

        if indice < len(EMNIST_CLASSES):

            clase = EMNIST_CLASSES[indice]

        else:

            clase = f"Clase {indice}"

        probabilidad = (
            probabilidades[indice] * 100
        )

        print(
            f"{posicion}. "
            f"{clase}: "
            f"{probabilidad:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Predicción de imágenes EMNIST"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen JPG o PNG"
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Etiqueta real de la imagen"
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("EMNIST - PREDICCIÓN DE IMAGEN")
    print("=" * 60)

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    modelo = cargar_modelo()

    # --------------------------------------------------------
    # Imagen
    # --------------------------------------------------------

    imagen, ruta_procesada = preparar_imagen(
        args.image
    )

    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    (
        indice,
        clase,
        confianza,
        probabilidades
    ) = realizar_prediccion(
        modelo,
        imagen
    )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    mostrar_resultado(
        args.image,
        imagen,
        indice,
        clase,
        confianza,
        ruta_procesada,
        args.label
    )

    # --------------------------------------------------------
    # Top 5
    # --------------------------------------------------------

    mostrar_top_predicciones(
        probabilidades
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
