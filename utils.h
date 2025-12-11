#ifndef UTILS_H
#define UTILS_H

#include "Matrix.h"
#include <string>
#include <vector>

std::vector<int> get_predictions(const Matrix& A2);

float get_accuracy(const std::vector<int>& predictions, const std::vector<int>& Y);

std::vector<int> read_mnist_labels(const std::string& full_path);
Matrix read_mnist_images(const std::string& full_path);

void print_image(const Matrix& data, int index);


#endif