#include "NeuralNetwork.h"
#include <fstream>
#include <iostream>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size) :
    W1(Matrix::random(hidden_size, input_size)),
    b1(Matrix(hidden_size, 1)),
    W2(Matrix::random(output_size, hidden_size)),
    b2(Matrix(output_size, 1))
    {

    }

NeuralNetwork::Cache NeuralNetwork::forward(const Matrix& X) {

    Matrix Z1 = W1.dot(X) + b1;
    Matrix A1 = relu(Z1);
    Matrix Z2 = W2.dot(A1) + b2;
    Matrix A2 = softmax(Z2);

    return {Z1, A1, Z2, A2};
}

void NeuralNetwork::backward(const Matrix& X, const std::vector<int>& Y, Cache cache) {
    int m = X.cols;

    Matrix one_hot_Y = one_hot(Y, 10);

    Matrix dZ2 = cache.A2  - one_hot_Y;

    Matrix dW2 = dZ2.dot(cache.A1.T()) * (1.0f / m);

    Matrix db2 = dZ2.sum_axis1() * (1.0f / m);

    Matrix dZ1 = W2.T().dot(dZ2) * relu_derivative(cache.Z1);

    Matrix dW1 = dZ1.dot(X.T())  * (1.0f / m);

    Matrix db1 = dZ1.sum_axis1() * (1.0f / m);

    W1 = W1 - (dW1 * learning_rate);
    b1 = b1 - (db1 * learning_rate);
    W2 = W2 - (dW2 * learning_rate);
    b2 = b2 - (db2 * learning_rate);

}

void NeuralNetwork::save(const std::string& filename) {
    std::ofstream output(filename, std::ios::binary);

    if (!output.is_open()) throw std::runtime_error("could not open file for saving");

    auto write_matrix  = [&] (const Matrix& m) {
        output.write(reinterpret_cast<const char *> (&m.rows), 4);
        output.write(reinterpret_cast<const char *> (&m.cols), 4);
        output.write(reinterpret_cast<const char *> (m.data.data()), m.data.size() * sizeof(float));
    };

    write_matrix(W1);
    write_matrix(W2);
    write_matrix(b1);
    write_matrix(b2);

}

void NeuralNetwork::load(const std::string& filename) {

    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) throw std::runtime_error("could not open file for loading");

    auto read_matrix = [&](Matrix& m) {
        input.read(reinterpret_cast<char*> (&m.rows), 4);
        input.read(reinterpret_cast<char*> (&m.cols), 4);
        m.data.resize(m.rows * m.cols);
        input.read(reinterpret_cast<char *> (m.data.data()), m.data.size() * sizeof(float));
    };

    read_matrix(W1);
    read_matrix(W2);
    read_matrix(b1);
    read_matrix(b2);
    
}


    
    


