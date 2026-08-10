import tensorflow_datasets as tfds

print("=" * 60)
print("MAPEO REAL DE EMNIST BALANCED")
print("=" * 60)

# Cargar información del dataset
builder = tfds.builder("emnist/balanced")
builder.download_and_prepare()

# Obtener los datos de entrenamiento
ds = builder.as_dataset(
    split="train",
    as_supervised=False
)

print("\nAnalizando etiquetas del dataset...\n")

# Buscar ejemplos de cada clase
ejemplos = {}

for ejemplo in ds:

    label = int(ejemplo["label"].numpy())

    if label not in ejemplos:
        ejemplos[label] = ejemplo

    if len(ejemplos) == 47:
        break

print("Clases encontradas:")
print()

for label in sorted(ejemplos.keys()):
    print(f"Índice {label}: existe en EMNIST")

print()
print("=" * 60)
print("IMPORTANTE")
print("=" * 60)
print(
    "TensorFlow Datasets está exponiendo las etiquetas "
    "como índices numéricos."
)
print(
    "Para convertirlas a caracteres debemos utilizar "
    "el mapeo oficial de EMNIST Balanced."
)
