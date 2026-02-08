<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { learningRateStore, allDigitsNetworkStore } from '../../stores';
	import RangeSlider from 'svelte-range-slider-pips';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import { testNetwork } from '$lib/NetworkTesting';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import { makeTopNLinksFilter, neighborsFilter } from '$lib/LinkFilters';
	import type { Neuron } from '$lib/NetworkShape';

	const logger = new tslog.Logger({ name: 'all_digits' });

	let learningRates = [0];

	const networkStore = allDigitsNetworkStore;

	$: networkShape = $networkStore?.shape;
	$: classes = $networkStore?.shape.classes;
	$: weights = $networkStore?.tfModel.weights;

	let drawbox: DrawBox;
	let prediction: number[] | undefined;
	let activations: number[][] | undefined;
	const defaultLinkFilter = makeTopNLinksFilter(700);
	let linkFilter = defaultLinkFilter;
	let image: ImageData | undefined;

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		learningRateStore.load().then((value) => {
			learningRates = [value];
		});
		isLoading = false;
	});

	onDestroy(async () => {
		learningRateStore.set(learningRates[0]);
	});

	function handleNeuronSelected(event: { detail: Neuron | null }) {
		const neuron = event.detail;
		if (neuron == null) {
			linkFilter = defaultLinkFilter;
		} else {
			linkFilter = neighborsFilter(neuron);
		}
	}

	function handleDrawnImage(event: { detail: { image: ImageData } }) {
		image = event.detail.image;
		predict_image();
	}

	function predict_image() {
		if (image) {
			activations = calculateActivations(image);
			prediction = activations[activations.length - 1];
			logger.debug('tf.memory() ', tf.memory());
		} else {
			activations = undefined;
			prediction = undefined;
		}
	}

	function calculateActivations(image: ImageData): number[][] {
		return tf.tidy(() => {
			const pixels = tf.browser.fromPixels(image, 1);

			// From: https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/index.js
			const processedImage = tf
				.reshape(pixels, [1, 28 * 28])
				.toFloat()
				.div(255);

			const activationsTensor = $networkStore.featureModel.predict(processedImage) as tf.Tensor[];
			return [processedImage, ...activationsTensor].map((x) =>
				tf.squeeze(x).arraySync()
			) as number[][];
		});
	}

	async function trainOnData(trainDataSize: number, batchSize: number) {
		logger.debug('before generating train data: tf.memory()', tf.memory());
		const [trainXs, trainYs] = tf.tidy(() => {
			const d = $networkStore.nextTrainBatch(trainDataSize);
			return [d.xs.reshape([trainDataSize, 28 * 28]), d.ys];
		});
		const validationDataSize = Math.ceil(trainDataSize / 20);
		const [valXs, valYs] = tf.tidy(() => {
			const d = $networkStore.nextTrainBatch(validationDataSize);
			return [d.xs.reshape([validationDataSize, 28 * 28]), d.ys];
		});
		logger.debug('after generating train data: tf.memory()', tf.memory());
		train(trainXs, trainYs, valXs, valYs, batchSize, learningRates[0]);
	}

	async function train(
		trainXs: tf.Tensor2D,
		trainYs: tf.Tensor2D,
		valXs: tf.Tensor2D | null,
		valYs: tf.Tensor2D | null,
		batchSize: number,
		learningRate: number
	) {
		const networkUnderTraining = $networkStore;
		const optimizer = networkUnderTraining.tfModel.optimizer as tf.SGDOptimizer;
		optimizer.setLearningRate(learningRate);

		function onBatchBegin(_batch: number, _logs?: tf.Logs) {
			predict_image();
		}

		function onBatchEnd(batch: number, logs?: tf.Logs) {
			logger.debug('end batch:', batch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: logs?.size || 0,
				finalAccuracy: logs?.acc
			});
			const testResult = testNetwork(networkUnderTraining, classes?.length * 50);
			networkUnderTraining.stats.test = testResult;
			networkStore.update((n) => n); // Notify subscribers
		}

		function onEpochEnd(epoch: number, logs?: tf.Logs) {
			logger.debug('end epoch:', epoch, '. logs:', logs);
			if (logs?.val_acc) {
				networkUnderTraining.trainingRoundDone({
					samplesSeen: 0,
					finalAccuracy: logs.val_acc
				});
				networkStore.update((n) => n); // Notify subscribers
			}
		}

		function onTrainEnd(_logs?: tf.Logs) {
			logger.debug('onTrain end : tf.memory()', tf.memory());
			tf.dispose(trainXs);
			tf.dispose(trainYs);
			if (valXs) {
				tf.dispose(valXs);
			}
			if (valYs) {
				tf.dispose(valYs);
			}
			logger.debug('after disposing: tf.memory()', tf.memory());
		}

		logger.debug('Before fit: tf.memory()', tf.memory());

		// If this fails because there is already another fit running
		// Then the 4 tensors get leaked (because the cleanup occurs in
		// onTrainEnd, which is never called)
		const params = {
			epochs: 1,
			batchSize: batchSize,
			shuffle: true,
			callbacks: { onBatchBegin, onBatchEnd, onEpochEnd, onTrainEnd }
		};
		if (valXs && valYs) {
			params['validationData'] = [valXs, valYs];
		}

		return networkUnderTraining.tfModel.fit(trainXs, trainYs, params);
	}

	async function train100() {
		trainOnData(100, 50);
	}

	async function train1000() {
		trainOnData(1000, 50);
	}

	async function train5000() {
		trainOnData(5000, 50);
	}

	function resetModel() {
		networkStore.reload();
		predict_image();
	}

	function clear() {
		drawbox.clear();
		image = undefined;
		predict_image();
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-9 gap-4">
		<div class="col-span-2">
			<h4 class="text-xl mb-2">Dessiner un chiffre</h4>
			<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
			<button class="btn btn-outline btn-primary mt-4" disabled={!image} on:click={clear}
				>Effacer</button
			>

			<div class="divider"></div>

			<h4 class="text-xl">Apprentissage</h4>

			<ul class="menu py-4">
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={train100}>
						Entraîner avec 100 images
					</button>
				</li>
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={train1000}>
						Entraîner avec 1'000 images
					</button>
				</li>
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={train5000}>
						Entraîner avec 5'000 images
					</button>
				</li>
			</ul>

			<div class="text-l mb-2">Taux d'apprentissage</div>
			<RangeSlider
				bind:values={learningRates}
				min={0}
				max={1}
				step={0.2}
				pips
				all="label"
				springValues={{ stiffness: 0.2, damping: 0.7 }}
			/>
		</div>

		<div class="col-span-5">
			<NetworkGraph
				{networkShape}
				{weights}
				{activations}
				{linkFilter}
				on:neuronSelected={handleNeuronSelected}
			/>
		</div>
		<div class="col-span-2">
			<NetworkStats stats={$networkStore.stats} />
			<div class="m-6" />
			<button
				class="btn btn-outline btn-error"
				disabled={$networkStore.stats.samplesSeen == 0}
				on:click={resetModel}
			>
				Réinitialiser le réseau
			</button>
		</div>
	</div>
{/if}
