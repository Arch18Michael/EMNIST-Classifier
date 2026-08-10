import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

print("=" * 60)
print("EMNIST - ENTRENAMIENTO")
print("=" * 60)

print("TensorFlow:", tf.__version__)

# ============================================================
# CARGAR EMNIST
# ============================================================

print("\nCargando EMNIST...")

(ds_train, ds_test), ds_info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

num_classes = ds_info.features["label"].num_classes

print("Número de clases:", num_classes)
print("Entrenamiento:", ds_info.splits["train"].num_examples)
print("Prueba:", ds_info.splits["test"].num_examples)


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preprocess(image, label):

    image = tf.cast(image, tf.float32)

    # Normalización 0-255 -> 0-1
    image = image / 255.0

    return image, label


ds_train = ds_train.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

ds_test = ds_test.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# BATCH
# ============================================================

BATCH_SIZE = 128

ds_train = ds_train.shuffle(10000).batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

ds_test = ds_test.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# MODELO
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# MOSTRAR MODELO
# ============================================================

model.summary()


# ============================================================
# COMPILAR
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\nComenzando entrenamiento...")

history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=10
)


# ============================================================
# EVALUACIÓN
# ============================================================

print("\nEvaluando modelo...")

loss, accuracy = model.evaluate(ds_test)

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# GUARDAR MODELO
# ============================================================

os.makedirs("modelos", exist_ok=True)

model.save(
    "modelos/emnist_model.keras"
)

print("\nModelo guardado en:")
print("modelos/emnist_model.keras")
