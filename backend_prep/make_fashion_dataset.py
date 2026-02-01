#!/usr/bin/env python

import numpy as np
from keras import datasets, utils


def load():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    images = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))
    return (images, labels)


def write_dataset(images, labels, dataset_name):
    def label_to_activation(label):
        activation = [0] * 10
        activation[label] = 1
        return activation

    activations = [label_to_activation(label) for label in labels]
    activations = np.array(activations, dtype=np.uint8)
    activations.tofile(f"build/{dataset_name}_labels_uint8")

    flattened_images = np.reshape(images, [-1, 28 * 28, 1])
    utils.save_img(f"build/{dataset_name}_images.png", flattened_images)


if __name__ == "__main__":
    images, labels = load()
    print(
        "number of images",
        len(images),
        "\n",
        "number of labels",
        len(labels),
    )
    write_dataset(images, labels, "fashion")
