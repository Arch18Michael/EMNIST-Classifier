import tensorflow_datasets as tfds

ds, info = tfds.load(
    "emnist/balanced",
    split="train",
    with_info=True
)

print("Número de clases:", info.features["label"].num_classes)
print("\nClases reales de EMNIST:")
print(info.features["label"].names)
