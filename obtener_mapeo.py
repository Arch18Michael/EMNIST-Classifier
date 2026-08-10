import tensorflow_datasets as tfds

_, info = tfds.load(
    "emnist/balanced",
    split="train",
    with_info=True
)

label_feature = info.features["label"]

print("Número de clases:", label_feature.num_classes)
print("\nNombres de las clases:")
print(label_feature.names)

print("\nValores de las clases:")
for i, nombre in enumerate(label_feature.names):
    print(f"{i}: {nombre}")
