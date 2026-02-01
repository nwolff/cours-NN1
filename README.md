A web application to experiment with a neural network that recognizes handwritten digits : training, predicting, evaluating the precision.

Deployed automatically when branch main is pushed, to:

http://nn.nwolff.info/

Usage statistics collected with umami.js

The python backend is only needed if you want to train new/different models or generate new datasets for use in the frontend.

# Tooling

Requires an installation of nodejs

## Installing dependencies

    npm install

## Updating dependencies

    npm update

## Developing

Running the app under development, with automatic reload:

    npm run dev -- --open

Automatically formatting:

    npm run format

Type-checking:

    npm run check

There are currently some typing errors, many because we extract data from tensors that are very generically typed.

## Verifying the production build

Sveltekit has a server-side rendering capability to optimize page loads.
We want to build a single page app that will be served statically, so we have to disable server-side rendering for each of our routes.
This step makes sure we haven't forgotten anything.

    npm run build
    npm run preview -- --open

## Deploying to GitHub pages manually

(The app is currently automatically deployed with github actions when the main branch is pushed)

This will run a full build and deploy

    npm run gh-pages

# Refs

https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Deep-Dive-Into-Learning-Curves-in-Machine-Learning--Vmlldzo0NjA1ODY0

https://medium.com/duke-ai-society-blog/training-neural-networks-for-binary-classification-identifying-types-of-breast-cancer-keras-in-r-b38fb26a500c

https://artemoppermann.com/activation-functions-in-deep-learning-sigmoid-tanh-relu/

https://www.tensorflow.org/tutorials/keras/keras_tuner

https://medium.com/@chamara95.eng/neural-network-example-using-fashion-mnist-dataset-c19b48c86cf1

# Technical

https://github.com/idris-maps/svelte-parts/blob/master/src/lib/DropFile.svelte

https://blog.filestack.com/how-to-read-uploaded-file-content-in-javascript/

https://dev.to/dailydevtips1/vanilla-javascript-canvas-images-to-black-and-white-mpe

https://plotly.com/javascript/plotlyjs-events/

https://github.com/plotly/plotly.js/blob/master/src/plot_api/plot_config.js
