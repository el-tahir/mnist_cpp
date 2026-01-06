/**
 * @file Matrix.h
 * @brief Matrix class and mathematical operations for neural network computations.
 * 
 * Provides a 2D matrix container with support for common linear algebra operations,
 * AVX-optimized matrix multiplication, and neural network activation functions.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>

/**
 * @class Matrix
 * @brief A 2D matrix class for neural network computations.
 * 
 * Stores data in row-major order and provides operations for matrix arithmetic,
 * transposition, and dot products. Supports AVX vectorization for optimized
 * matrix multiplication on compatible systems.
 */
class Matrix {
public:    
    std::vector<float> data;  ///< Flat storage for matrix elements in row-major order.
    int rows;                 ///< Number of rows in the matrix.
    int cols;                 ///< Number of columns in the matrix.

    /**
     * @brief Constructs a matrix with the specified dimensions, initialized to zeros.
     * @param r Number of rows.
     * @param c Number of columns.
     */
    Matrix(int r, int c);

    /**
     * @brief Creates a matrix with random values uniformly distributed in [-1, 1].
     * @param r Number of rows.
     * @param c Number of columns.
     * @return A new Matrix with random values.
     */
    static Matrix random(int r, int c);

    /**
     * @brief Accesses an element at the specified position (const version).
     * @param r Row index (0-indexed).
     * @param c Column index (0-indexed).
     * @return The value at position (r, c).
     */
    float operator()(int r, int c) const;

    /**
     * @brief Accesses an element at the specified position (mutable version).
     * @param r Row index (0-indexed).
     * @param c Column index (0-indexed).
     * @return Reference to the value at position (r, c).
     */
    float& operator()(int r, int c);

    /**
     * @brief Computes the matrix dot product (matrix multiplication).
     * @param other The right-hand side matrix.
     * @return The product matrix with dimensions (this->rows x other.cols).
     * @throws std::invalid_argument If matrix dimensions are incompatible.
     * @note Uses AVX vectorization when available for improved performance.
     */
    Matrix dot(const Matrix& other) const;

    /**
     * @brief Returns the transpose of this matrix.
     * @return A new Matrix with dimensions (cols x rows).
     */
    Matrix T() const;

    /**
     * @brief Sums elements along axis 1 (columns), reducing to a column vector.
     * @return A column vector (rows x 1) containing row sums.
     */
    Matrix sum_axis1() const;

    /**
     * @brief Element-wise subtraction of two matrices.
     * @param other The matrix to subtract.
     * @return A new Matrix containing the difference.
     * @throws std::invalid_argument If matrix dimensions do not match.
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * @brief Element-wise addition of two matrices with broadcasting support.
     * @param other The matrix to add.
     * @return A new Matrix containing the sum.
     * @throws std::invalid_argument If matrix dimensions are incompatible.
     * @note Supports broadcasting when other is a column vector (rows x 1).
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * @brief Scalar multiplication of the matrix.
     * @param scalar The scalar value to multiply.
     * @return A new Matrix with all elements scaled.
     */
    Matrix operator*(float scalar) const;

    /**
     * @brief Element-wise (Hadamard) multiplication of two matrices.
     * @param other The matrix to multiply element-wise.
     * @return A new Matrix containing the element-wise product.
     * @throws std::invalid_argument If matrix dimensions do not match.
     */
    Matrix operator*(const Matrix& other) const;

};


/**
 * @brief Applies the ReLU activation function element-wise.
 * @param Z Input matrix.
 * @return A new Matrix with ReLU applied: max(0, x) for each element.
 */
Matrix relu(const Matrix& Z);

/**
 * @brief Computes the derivative of ReLU element-wise.
 * @param Z Input matrix (pre-activation values).
 * @return A new Matrix with values 1.0 where Z > 0, else 0.0.
 */
Matrix relu_derivative(const Matrix& Z);

/**
 * @brief Applies the softmax function column-wise for numerical stability.
 * @param Z Input matrix where each column is a separate sample.
 * @return A new Matrix with softmax probabilities (columns sum to 1).
 */
Matrix softmax(const Matrix& Z);

/**
 * @brief Creates a one-hot encoded matrix from label indices.
 * @param Y Vector of label indices (0 to num_classes-1).
 * @param num_classes Number of classes.
 * @return A matrix of shape (num_classes x Y.size()) with one-hot encoding.
 */
Matrix one_hot(std::vector<int> Y, int num_classes);

#endif