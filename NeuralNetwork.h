/**
 * @file NeuralNetwork.h
 * @brief Simple two-layer neural network for MNIST digit classification.
 * 
 * Implements a fully-connected neural network with one hidden layer,
 * using ReLU activation and softmax output.
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"
#include <string>
#include <vector>

/**
 * @class NeuralNetwork
 * @brief A two-layer fully-connected neural network.
 * 
 * Architecture: Input -> Hidden (ReLU) -> Output (Softmax)
 * Trained using gradient descent with backpropagation.
 */
class NeuralNetwork {
public:
    Matrix W1, W2, b1, b2;  ///< Weight matrices and bias vectors for both layers.

    float learning_rate = 0.01f;  ///< Learning rate for gradient descent updates.

    /**
     * @brief Constructs a neural network with specified layer sizes.
     * @param input_size Number of input features (e.g., 784 for MNIST).
     * @param hidden_size Number of neurons in the hidden layer.
     * @param output_size Number of output classes (e.g., 10 for digits).
     */
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    
    /**
     * @struct Cache
     * @brief Stores intermediate values from the forward pass for backpropagation.
     */
    struct Cache {
        Matrix Z1, A1, Z2, A2;  ///< Pre-activations (Z) and activations (A) for each layer.
    };

    /**
     * @brief Performs forward propagation through the network.
     * @param X Input matrix of shape (num_features x batch_size).
     * @return Cache containing intermediate activations.
     */
    Cache forward(const Matrix& X);

    /**
     * @brief Performs backpropagation and updates weights using gradient descent.
     * @param X Input matrix of shape (num_features x batch_size).
     * @param Y Vector of ground truth labels for the batch.
     * @param cache Forward pass cache containing intermediate values.
     */
    void backward(const Matrix& X, const std::vector<int>& Y, Cache cache);

    /**
     * @brief Saves the network weights and biases to a binary file.
     * @param filename Path to the output file.
     * @throws std::runtime_error If the file cannot be opened.
     */
    void save(const std::string& filename);

    /**
     * @brief Loads network weights and biases from a binary file.
     * @param filename Path to the input file.
     * @throws std::runtime_error If the file cannot be opened.
     */
    void load(const std::string& filename);
    
};


#endif