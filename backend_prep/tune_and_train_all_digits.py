#!/usr/bin/env python

import keras_tuner
from keras import datasets, layers, losses, models, optimizers

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

IMAGE_SIZE = 28
# Normalize values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# For images we unroll the pixel grid
x_train = x_train[:2000, :]
y_train = y_train[:2000]
x_val = x_train[-2000:, :]
y_val = y_train[-2000:]


def build_model(hp):
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
            layers.Dense(
                hp.Choice("units", [24, 32, 40]),
                activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"]),
            ),
            layers.Dense(
                hp.Choice("units", [24, 32, 40]),
                activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"]),
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
    project_name="tune_add_digits_hyperparameters",
    overwrite=True,
    hyperband_iterations=10,
)

tuner.search_space_summary()

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]


best_model.summary()

best_model.compile(
    optimizer="adam",
    loss=losses.sparse_categorical_crossentropy,
    metrics=["accuracy"],
)

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

best_model.save("build/all_digits.keras")
