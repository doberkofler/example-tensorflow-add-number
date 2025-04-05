import * as tf from '@tensorflow/tfjs-node';
import * as readline from 'readline';

// Create readline interface for CLI input
const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
});

// Load the trained model from the JSON file
const loadModel = async (path: string) => {
	const model = await tf.loadLayersModel(`file://${path}/model.json`);
	console.log('Model loaded successfully!');
	return model;
};

// Function to make a prediction using the trained model
const predictSum = async (model: tf.LayersModel, a: number, b: number) => {
	const prediction = model.predict(tf.tensor2d([[a, b]])) as tf.Tensor;
	prediction.print();
	return prediction.dataSync()[0]; // Return the predicted sum
};

// Main function to run the CLI app
const main = async () => {
	// Load the model
	const model = await loadModel('./saved_model');

	// Prompt for user input
	rl.question('Enter two natural numbers to add (e.g., "5 7"): ', async (input) => {
		const numbers = input.split(' ').map(Number);
		if (numbers.length === 2) {
			const [a, b] = numbers;
			const sum = await predictSum(model, a, b);
			console.log(`The predicted sum of ${a} + ${b} is: ${sum}`);
		} else {
			console.log('Please enter exactly two natural numbers.');
		}
		rl.close();
	});
};

// Run the prediction process
main().catch(console.error);
