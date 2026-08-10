import tensorflow_datasets as tfds

print("=" * 60)
print("MAPEO OFICIAL DE EMNIST BALANCED")
print("=" * 60)

builder = tfds.builder("emnist/balanced")

builder.download_and_prepare()

info = builder.info

print()
print("Número de clases:", info.features["label"].num_classes)

print()
print("Nombres oficiales:")

nombres = info.features["label"].names

for i, nombre in enumerate(nombres):
    print(f"{i}: {nombre}")
