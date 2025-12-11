#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"
#include <string>
#include <vector>

class NeuralNetwork {
public:
    Matrix W1, W2, b1, b2;

    float learning_rate = 0.01f;

    NeuralNetwork(int input_size, int hidden_size, int output_size);
    
    
    struct Cache {
        Matrix Z1, A1, Z2, A2;
    };

    Cache forward(const Matrix& X);
    void backward(const Matrix& X, const std::vector<int>& Y, Cache cache);

    void save(const std::string& filename);
    void load(const std::string& filename);
    
};


#endif