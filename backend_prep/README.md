# Setting up dependencies

We use uv. All the requirements are in pyproject.toml

# Preparing datasets

## Zero one dataset
We want a trimmed-down version of the MNIST dataset with only zeroes and ones.

Because it's painful to do the trimming in the frontend (masking data is asynchronous and that mixes really badly with the manual memory management required by tensors),
we prepare that dataset here in python and then use it from the frontend.

    uv run make_zero_one_dataset.py
    cp build/zero_one* ../static

## Fashion
The original format it's in (gzipped binary files) would mean more work in the frontend.
We prefer preparing the data in the backend, because we get support from keras and because python is simpler.


    uv run make_fashion_dataset.py
    cp build/fashion* ../static


# Training

The idea is to show pre-trained models in the frontend, in order to validate/compare with the
model currently being trained.

## Training the models

The first model is very simple and trains quickly, the second one is much slower

    uv run tune_and_train_all_digits.py
    uv run train_lenet_all_digits.py

## Converting trained models for the frontend to use

     tensorflowjs_converter --input_format=keras build/all_digits.keras \
     ../static/all_digits

     tensorflowjs_converter --input_format=keras build/le_net.keras \
     ../static/le_net
