
package com.ugurcan195260063.main;


import java.util.Random;

class NeuralNetwork {
    int inputSize;
    int hiddenSize;
    int outputSize;
    double[][] weightsInputHidden;
    double[][] weightsHiddenOutput;
    double learningRate;

    // Constructor to initialize the neural network
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        // Randomly initializing weights
        weightsInputHidden = new double[inputSize][hiddenSize];
        weightsHiddenOutput = new double[hiddenSize][outputSize];

        Random rand = new Random();
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = rand.nextDouble() * 2 - 1;  // [-1, 1] aralığında
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenOutput[i][j] = rand.nextDouble() * 2 - 1;  // [-1, 1] aralığında
            }
        }
    }

    // Sigmoid activation function
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of the sigmoid function
    public double sigmoidDerivative(double x) {
        double sigmoidValue = sigmoid(x);
        return sigmoidValue * (1 - sigmoidValue);
    }

    // Feedforward process
    public double[] feedForward(double[] inputs) {
        double[] hiddenLayerOutputs = new double[hiddenSize];
        double[] finalOutputs = new double[outputSize];

        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * weightsInputHidden[j][i];
            }
            hiddenLayerOutputs[i] = sigmoid(sum);
        }

        for (int i = 0; i < outputSize; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenLayerOutputs[j] * weightsHiddenOutput[j][i];
            }
            finalOutputs[i] = sigmoid(sum);
        }

        return finalOutputs;
    }

    // Backpropagation process
    public void backpropagation(double[] inputs, double[] targets) {
        double[] hiddenLayerOutputs = new double[hiddenSize];
        double[] finalOutputs = feedForward(inputs);

        double[] outputErrors = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputErrors[i] = targets[i] - finalOutputs[i];
        }

        double[] hiddenErrors = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double error = 0;
            for (int j = 0; j < outputSize; j++) {
                error += outputErrors[j] * weightsHiddenOutput[i][j];
            }
            hiddenErrors[i] = error * sigmoidDerivative(hiddenLayerOutputs[i]);
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHiddenOutput[j][i] += learningRate * outputErrors[i] * hiddenLayerOutputs[j];
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputHidden[j][i] += learningRate * hiddenErrors[i] * inputs[j];
            }
        }
    }

    // Train the neural network
    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainingInputs.length; i++) {
                backpropagation(trainingInputs[i], trainingOutputs[i]);
            }
        }
    }
}

public class Main {

    // Normalization function
    public static double[] normalize(double[] data, double[] min, double[] max) {
        double[] normalized = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            normalized[i] = (data[i] - min[i]) / (max[i] - min[i]);
        }
        return normalized;
    }

    // Denormalization function
    public static double[] denormalize(double[] data, double[] min, double[] max) {
        double[] denormalized = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            denormalized[i] = data[i] * (max[i] - min[i]) + min[i];
        }
        return denormalized;
    }

    public static void main(String[] args) {
        // Training data for house price prediction (before normalization)
        double[][] inputs = {
            {120, 3, 1},  // 120 m^2, 3 room, zone 1 (Ankara)
            {200, 4, 2},  // 200 m^2, 4 room, zone 2 (Istanbul)
            {150, 3, 2},  // 150 m^2, 3 room, zone 2 (Istanbul)
            {80, 2, 1}    // 80 m^2, 2 room, zone 1 (Ankara)
        };

        double[][] outputs = {
            {300},  // 300.000 TL
            {500},  // 500.000 TL
            {400},  // 400.000 TL
            {150}   // 150.000 TL
        };

        // Initialize neural network (learning rate: 0.01)
        NeuralNetwork nn = new NeuralNetwork(3, 5, 1, 0.01);

        // Min and max values for normalization
        double[] minValues = {80, 2, 1};  // Min for each feature (m^2, rooms, zone)
        double[] maxValues = {200, 4, 2}; // Max for each feature

        // Normalize inputs
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = normalize(inputs[i], minValues, maxValues);
        }

        // Normalize outputs (for price between 150k to 500k TL, min = 150, max = 500)
        double[] minPrice = {150};
        double[] maxPrice = {500};

        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = normalize(outputs[i], minPrice, maxPrice);
        }

        // Train the neural network
        nn.train(inputs, outputs, 10000);

        // Test the neural network on new inputs
        double[][] testInputs = {
            {300, 5, 1},  // 100 m^2, 3 room, zone 1 (Istanbul)
            {150, 2, 2}   // 250 m^2, 4 room, zone 2 (Ankara)
        };

        // Normalize test inputs
        for (int i = 0; i < testInputs.length; i++) {
            testInputs[i] = normalize(testInputs[i], minValues, maxValues);
        }

        for(double[] input : testInputs){
            double[] output = nn.feedForward(input);
            String zoneName = Math.round(input[2]*(maxValues[2]-minValues[2])+minValues[2]) == 1 ? "Istanbul" : "Ankara";
            double[] predictedPrice = denormalize(output,minPrice,maxPrice);
            System.out.println("Input: "+(input[0] * (maxValues[0]-minValues[0])+minValues[0]) + "m^2, "
                    + (input[1]*(maxValues[1]-minValues[1]) + minValues[1]) + " room, zone "
                    + zoneName +
                    " - Predicted Price[Tahmin Edilen Fiyat]: " + predictedPrice[0] + " TL (Turkish Lira)"
            );
        }

        }
    }



