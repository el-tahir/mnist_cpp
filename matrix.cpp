#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdint>

class Matrix {
public:    
    std::vector<float> data;

    int rows;
    int cols;

    Matrix(int r, int c): rows(r), cols(c) {
        data.resize(rows * cols, 0.0f);
    }

    static Matrix random(int r, int c) {
        Matrix m(r, c);

        for (int i = 0; i < m.data.size(); i++)
        m.data[i] = ((rand() / float(RAND_MAX)) * 2.0f - 1.0f);

        return m;
    }

    float operator()(int r, int c) const {
        int index = (r* cols) + c;
        return data[index];

    }

    float& operator()(int r, int c) {
        int index = (r* cols) + c;
        return data[index];

    }

    Matrix dot(const Matrix& other) const {

        if (this->cols != other.rows) {
            throw std::invalid_argument("Shape mismatch");
        }
        Matrix result(this->rows, other.cols);
        for (int i = 0; i < this->rows; i++) { // cows of matrix A
            for (int j = 0; j < other.cols; j++) { // rows of matrix B
                float sum = 0.0f;
                for (int k = 0; k < this->cols; k++) { // common dimension
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    Matrix T() const {
        Matrix result(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (other.rows != this->rows || other.cols != this->cols) {
            throw std::invalid_argument("Shape mismatch");
        }

        Matrix result(rows, cols);

        for (size_t i = 0; i < this->data.size(); i++) {
                result.data[i] = this->data[i] - other.data[i];
        }

        return result;
    }

    Matrix operator+(const Matrix& other) const {

        Matrix result(rows, cols);

        if (this->rows == other.rows && this->cols == other.cols) { // standard addition
                
            for (size_t i = 0; i < this->data.size(); i++) {
                result.data[i] = this->data[i] + other.data[i];

            }

            return result;
        }

        if (this->rows == other.rows && other.cols == 1) { // broadasting
            for (size_t i = 0; i < this->rows; i++) {
                for (size_t j = 0; j < this->cols; j++) {
                    result(i, j) = (*this)(i, j) + other(i, 0);
                }
            }
            return result;
        }

        throw std::invalid_argument("Shape mismatch");
    }

    Matrix operator*(float scalar) const {
        Matrix result(rows, cols);

        for (size_t i = 0; i < this->data.size(); i++) {
            result.data[i] = scalar * this->data[i];
        }

        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (this->rows != other.rows || this->cols != other.cols) {
            throw std::invalid_argument("Shape mismathc");
        }

        Matrix result(rows, cols);

        for (size_t i = 0; i < this->data.size(); i++) {
            result.data[i] = this->data[i] * other.data[i];
        }

        return result;

    }
    Matrix sum_axis1() const {
        Matrix result(this->rows, 1);

        for (size_t i = 0; i < this->rows; i++) {
            float sum = 0;
            for (size_t j = 0; j < this->cols; j++) {
                sum += (*this)(i, j);
            }
            result(i, 0) = sum;
        }

        return result;

    }

};


Matrix relu(const Matrix& Z) {
    Matrix result(Z.rows, Z.cols);

    for (size_t i = 0; i < Z.data.size(); i++) {
        result.data[i] = std::max(Z.data[i], 0.0f);
    }

    return result;
}

Matrix relu_derivative(const Matrix& Z) {
    Matrix result(Z.rows, Z.cols);

    for (size_t i = 0; i < Z.data.size(); i++) {
        result.data[i] = (Z.data[i] > 0) ? 1.0f : 0.0f;
    }

    return result; 
}

Matrix softmax(const Matrix& Z){
    Matrix result(Z.rows, Z.cols);

    for (size_t j = 0; j < Z.cols; j++) {
        float maxi = Z(0, j);

        for (size_t i = 1; i < Z.rows; i++) {
            maxi = std::max(maxi, Z(i, j));
        }

        float sum_exp = 0.0f;

        for (size_t i = 0; i < Z.rows; i++) {
            float exp_val = std::exp(Z(i, j) - maxi);

            result(i, j) = exp_val;

            sum_exp += exp_val;
        }

        for (size_t i = 0; i < Z.rows; i++) {
            result(i, j) /= sum_exp;
        }
    }

    return result;
}


Matrix one_hot(std::vector<int> Y, int num_classes) {
    Matrix result(num_classes, Y.size());

    for (size_t i = 0; i < Y.size(); i++) {
        result(Y[i], i) = 1.0f;
    }

    return result;
}


class NeuralNetwork {
public:
    Matrix W1, W2, b1, b2;

    float learning_rate = 0.01f;

    NeuralNetwork(int input_size, int hidden_size, int output_size) :
        W1(Matrix::random(hidden_size, input_size)),
        b1(Matrix(hidden_size, 1)),
        W2(Matrix::random(output_size, hidden_size)),
        b2(Matrix(output_size, 1))
        {

        }
    
    
    struct Cache {
        Matrix Z1, A1, Z2, A2;
    };

    Cache forward(const Matrix& X) {

        Matrix Z1 = W1.dot(X) + b1;
        Matrix A1 = relu(Z1);
        Matrix Z2 = W2.dot(A1) + b2;
        Matrix A2 = softmax(Z2);

        return {Z1, A1, Z2, A2};
    }

    void backward(const Matrix& X, const std::vector<int>& Y, Cache cache) {
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


};

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

int main() {
    Matrix train_images = read_mnist_images("train-images-idx3-ubyte");
    std::vector<int> train_labels = read_mnist_labels("train-labels-idx1-ubyte");

    Matrix test_images = read_mnist_images("t10k-images-idx3-ubyte");
    std::vector<int> test_labels = read_mnist_labels("t10k-labels-idx1-ubyte");


    NeuralNetwork nn(784, 128, 10);

    int epochs = 20;

    int batch_size = 64;

    int num_samples = train_images.rows;
    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {

        for (int batch = 0; batch < num_batches; batch++) {

            int start = batch * batch_size;
            int end = std::min(start + batch_size, num_samples);
            int current_batch_size = end - start;

            Matrix X_batch (current_batch_size, 784);

            for (int i = 0; i < current_batch_size; i++) {
                for (int j = 0; j < 784; j++) {
                    X_batch(i, j) = train_images(start + i, j);
                }
            }

            std::vector<int> Y_batch;
            Y_batch.reserve(current_batch_size);

            for (int i = start; i < end; i++) {
                Y_batch.push_back(train_labels[i]);
            }

            Matrix X_batch_T = X_batch.T();
            auto cache =  nn.forward(X_batch_T);
            nn.backward(X_batch_T, Y_batch, cache);

        }

        auto cache = nn.forward(test_images.T());
        auto predictions = get_predictions(cache.A2);
        float acc = get_accuracy(predictions, test_labels);

        std::cout << "Epoch " <<  epoch << " accuracy: " << acc << std::endl;

    }




    return 0;
}


