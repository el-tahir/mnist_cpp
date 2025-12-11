#include "utils.h"
#include <fstream>
#include <algorithm>
#include <cstdint>

std::vector<int> get_predictions(const Matrix& A2) {
    // shape (10 x batch_size)

    std::vector<int> result;
    result.reserve(A2.cols);

    for (int j = 0; j < A2.cols; j++) {
        float maxi = A2(0, j);
        int maxi_index = 0;


        for (int i = 0; i < A2.rows; i++) {
            if (A2(i, j) > maxi) {
                maxi = A2(i, j);
                maxi_index = i;
            }
        }

        result.push_back(maxi_index);
    }

    return result;
}

float get_accuracy(const std::vector<int>& predictions, const std::vector<int>& Y) {
    if (predictions.size() != Y.size()) throw std::invalid_argument("Shape mismatch");

    int count = 0;

    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i] == Y[i]) count++;
    }

    return count / float(predictions.size());
}


constexpr uint32_t swap_endian(uint32_t val) {
    return ((val & 0xFF000000) >> 24) |
        ((val & 0x00FF0000) >> 8) |
        ((val & 0x0000FF00) << 8) |
        ((val & 0x000000FF) << 24);
}

std::vector<int> read_mnist_labels(const std::string& full_path) {
    std::ifstream label_file(full_path, std::ios::binary);

    uint32_t magic;
    uint32_t num_items;

    label_file.read(reinterpret_cast<char*> (&magic), 4);
    magic = swap_endian(magic);

    if (magic != 2049) throw std::runtime_error("Invalid label file");

    label_file.read(reinterpret_cast<char*> (&num_items), 4);
    num_items = swap_endian(num_items);

    std::vector<int> result;
    result.reserve(num_items);

    for (int i = 0; i < num_items; i++) {
        uint8_t label;
        label_file.read(reinterpret_cast<char*>(&label), 1);

        result.push_back(label);

    }

    return result;
}

Matrix read_mnist_images(const std::string& full_path) {

    std::ifstream images_file(full_path, std::ios::binary);

    uint32_t magic;
    uint32_t num_images;
    uint32_t rows;
    uint32_t cols;

    images_file.read(reinterpret_cast<char*> (&magic), 4);
    magic = swap_endian(magic);

    if (magic != 2051) throw std::runtime_error("invalid images file");

    images_file.read(reinterpret_cast<char*> (& num_images), 4);
    num_images = swap_endian(num_images);

    images_file.read(reinterpret_cast<char*> (&rows), 4);
    rows = swap_endian(rows);

    images_file.read(reinterpret_cast<char*> (&cols), 4);
    cols = swap_endian(cols);

    Matrix images(num_images, rows * cols);

    for (int n = 0; n < num_images; n++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                uint8_t pixel;
                images_file.read(reinterpret_cast<char*> (&pixel), 1);
                images(n, i * cols + j) = float(pixel) / 255.0f;
            }
        }
    }

    return images;
    
}

void print_image(const Matrix& data, int index) {
    std::cout << "image index: " << index << std::endl;

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            float pixel = data(index, i * 28 + j);
            if (pixel > 0.8) std::cout << "@";
            else if (pixel > 0.5) std::cout << ".";
            else std::cout << " ";
        }
        std::cout << std::endl;
    }
}