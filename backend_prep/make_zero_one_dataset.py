#!/usr/bin/env python

import numpy as np
from keras import datasets, utils


def load():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    images = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))
    return (images, labels)


def keep_zeros_and_ones(images_and_labels):
    images, labels = images_and_labels
    label_to_mask = lambda label: label <= 1
    mask = label_to_mask(labels)
    return (images[mask], labels[mask])


def write_dataset(images, labels, dataset_name):
    def label_to_activation(label):
        if label == 0:
            return [1, 0]
        elif label == 1:
            return [0, 1]
        else:
            raise Exception("Unexpected label")

    activations = [label_to_activation(label) for label in labels]
    activations = np.array(activations, dtype=np.uint8)
    activations.tofile(f"build/{dataset_name}_labels_uint8")

    flattened_images = np.reshape(images, [-1, 28 * 28, 1])
    utils.save_img(f"build/{dataset_name}_images.png", flattened_images)


if __name__ == "__main__":
    images_and_labels = load()
    images, labels = keep_zeros_and_ones(images_and_labels)
    print(
        "number of images",
        len(images),
        "\n",
        "number of labels",
        len(labels),
    )
    write_dataset(images, labels, "zero_one")
