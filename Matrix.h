#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>

class Matrix {
public:    
    std::vector<float> data;
    int rows;
    int cols;

    Matrix(int r, int c);

    static Matrix random(int r, int c);

    float operator()(int r, int c) const;
    float& operator()(int r, int c);

    Matrix dot(const Matrix& other) const;
    Matrix T() const;
    Matrix sum_axis1() const;

    Matrix operator-(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(float scalar) const;
    Matrix operator*(const Matrix& other) const;

};


Matrix relu(const Matrix& Z);
Matrix relu_derivative(const Matrix& Z);
Matrix softmax(const Matrix& Z);
Matrix one_hot(std::vector<int> Y, int num_classes);

#endif