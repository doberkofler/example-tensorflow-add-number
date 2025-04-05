import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'node:fs';
import * as path from 'node:path';

// Function to create a simple model
const createModel = () => {
	const model = tf.sequential();
	model.add(tf.layers.dense({units: 1, inputShape: [2]})); // Two input numbers, one output number
	model.compile({
		optimizer: tf.train.adam(0.01),
		loss: 'meanSquaredError',
	});
	return model;
};

// Function to generate training data (pairs of natural numbers and their sums)
const generateTrainingData = (numExamples: number) => {
	const inputs: number[][] = [];
	const outputs: number[][] = [];
	for (let i = 0; i < numExamples; i++) {
		const a = Math.floor(Math.random() * 100); // Random natural number between 0 and 99
		const b = Math.floor(Math.random() * 100); // Random natural number between 0 and 99
		inputs.push([a, b]);
		outputs.push([a + b]);
	}
	return {inputs: tf.tensor2d(inputs), outputs: tf.tensor2d(outputs)};
};

// Function to train the model
const trainModel = async (model: tf.LayersModel, inputs: tf.Tensor, outputs: tf.Tensor) => {
	await model.fit(inputs, outputs, {
		epochs: 100,
		batchSize: 32,
		verbose: 0,
	});
	console.log('Model training complete.');
};

// Save the model to a JSON file
const saveModel = async (model: tf.LayersModel) => {
	// Save paths
	const modelSavePath = path.resolve('./saved_model');

	try {
		// Ensure the directory exists
		if (!fs.existsSync(modelSavePath)) {
			fs.mkdirSync(modelSavePath, {recursive: true});
		}

		// Save the model
		await model.save(`file://${modelSavePath}`);

		console.log('Model saved successfully');
	} catch (error) {
		console.error('Error saving model:', error);
	}
};

// Main function
const main = async () => {
	const model = createModel();
	const {inputs, outputs} = generateTrainingData(10000); // Train on 10000 examples
	await trainModel(model, inputs, outputs);
	await saveModel(model); // Save model to "trained-model" folder
};

// Run the training process
main().catch(console.error);
