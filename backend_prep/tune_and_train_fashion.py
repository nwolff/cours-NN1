#!/usr/bin/env python

import keras_tuner
from keras import layers, losses, models, optimizers

from make_fashion_dataset import load

images, labels = load()
# normalize values between 0 and 1
images = images / 255
IMAGE_SIZE = 28

# Fashion dataset has 70'000 images
# 0 -> 58'000 Train
# 58'000 -> 60'000 Validate
# 60'000 -> 70'000 Test

# For images we unroll the pixel grid
x_train = images[:-12_000, :]
y_train = labels[:-12_000]
x_val = images[-12_000:-10_000, :]
y_val = labels[-12_000:-10_000]
x_test = images[-10_000:, :]
y_test = labels[-10_000:]


def build_model(hp):
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
            layers.Dense(
                hp.Choice("units", [32, 40, 50]),
                activation="relu",
            ),
            layers.Dense(
                hp.Choice("units", [32, 40, 50]),
                activation="relu",
            ),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


tuner = keras_tuner.Hyperband(
    build_model,
    objective="loss",
    directory="build",
    project_name="tune_fashion_hyperparameters",
    overwrite=True,
    hyperband_iterations=30,
)

tuner.search_space_summary()

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

best_model = tuner.get_best_models()[0]

best_model.summary()

history = best_model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=40,
    validation_data=(x_val, y_val),
)

print("")
print("TESTING :")
scores = best_model.evaluate(x_test, y_test, verbose=0)
print("Loss: %.2f%%" % (scores[0] * 100))
print("Accuracy: %.2f%%" % (scores[1] * 100))

best_model.save("build/fashion.keras")
