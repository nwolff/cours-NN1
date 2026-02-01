#!/usr/bin/env python

import keras_tuner
from keras import layers, losses, models, optimizers

from make_zero_one_dataset import keep_zeros_and_ones, load

images, labels = keep_zeros_and_ones(load())
images = images / 255
IMAGE_SIZE = 28


#  Finding the ratios for the ZeroOne dataset, based on what we did for
#  the full dataset:
#
#           Full dataset    ZeroOne dataset
# Train + val   60'000          14'780
# Train         58'000          14'280
# Validate       2'000             500
# Test          10'000           2'500

x_train = images[:-3000, :]
y_train = labels[:-3000]
x_val = images[-3000:-2500, :]
y_val = labels[-3000:-2500]
x_test = images[-2500:, :]
y_test = labels[-2500:]


def build_model(hp):
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(
                hp.Choice("units", [16, 20, 24, 28, 32]),
                activation=hp.Choice(
                    "activation", ["relu", "leaky_relu", "relu6", "tanh", "sigmoid"]
                ),
            ),
            layers.Dense(2, activation="softmax"),
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
    project_name="tune_zero_one_hyperparameters",
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

best_model.save("build/zero_one.keras")
