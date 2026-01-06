/**
 * @file utils.h
 * @brief Utility functions for MNIST data loading, prediction, and visualization.
 */

#ifndef UTILS_H
#define UTILS_H

#include "Matrix.h"
#include <string>
#include <vector>

/**
 * @brief Extracts predicted class labels from network output.
 * @param A2 Output activations matrix of shape (num_classes x batch_size).
 * @return Vector of predicted class indices (argmax per column).
 */
std::vector<int> get_predictions(const Matrix& A2);

/**
 * @brief Computes classification accuracy.
 * @param predictions Vector of predicted class labels.
 * @param Y Vector of ground truth labels.
 * @return Accuracy as a float in range [0.0, 1.0].
 * @throws std::invalid_argument If vector sizes do not match.
 */
float get_accuracy(const std::vector<int>& predictions, const std::vector<int>& Y);

/**
 * @brief Reads MNIST label file in IDX format.
 * @param full_path Path to the MNIST labels file (e.g., "train-labels-idx1-ubyte").
 * @return Vector of integer labels.
 * @throws std::runtime_error If the file format is invalid.
 */
std::vector<int> read_mnist_labels(const std::string& full_path);

/**
 * @brief Reads MNIST image file in IDX format.
 * @param full_path Path to the MNIST images file (e.g., "train-images-idx3-ubyte").
 * @return Matrix of shape (num_images x 784) with normalized pixel values [0, 1].
 * @throws std::runtime_error If the file format is invalid.
 */
Matrix read_mnist_images(const std::string& full_path);

/**
 * @brief Prints an ASCII art representation of an MNIST image to stdout.
 * @param data Image dataset matrix of shape (num_images x 784).
 * @param index Index of the image to print.
 */
void print_image(const Matrix& data, int index);


#endif