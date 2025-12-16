#include "Matrix.h"
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define USE_AVX 1
    #pragma message("AVX support found. Training will be fast")
    #include <immintrin.h>
#else
    #define USE_AVX 0
    #pragma message("AVX not found. Training will be slower...")
#endif

Matrix::Matrix(int r, int c): rows(r), cols(c) {
    data.resize(rows * cols, 0.0f);
}

Matrix Matrix::random(int r, int c) {
    Matrix m(r, c);

    for (int i = 0; i < m.data.size(); i++)
    m.data[i] = ((rand() / float(RAND_MAX)) * 2.0f - 1.0f);

    return m;
}

float Matrix::operator()(int r, int c) const {
    int index = (r* cols) + c;
    return data[index];

}

float& Matrix::operator()(int r, int c) {
    int index = (r* cols) + c;
    return data[index];

}

Matrix Matrix::dot(const Matrix& other) const {

    if (this->cols != other.rows) {
        throw std::invalid_argument("Shape mismatch");
    }
    Matrix result(this->rows, other.cols);

    #if USE_AVX
        for (int i = 0; i < this->rows; i++) { 
            for (int k = 0; k < this->cols; k++) {
                
                float a_val = (*this)(i, k);

                __m256 vec_a = _mm256_set1_ps(a_val); // [a, a, a, a, a, a, a, a]
                
                int j = 0;

                for (; j <= other.cols - 8; j += 8) {
                    __m256 vec_c = _mm256_loadu_ps(&result.data[i * other.cols + j]);
                    __m256 vec_b = _mm256_loadu_ps(&other.data[k * other.cols + j]);

                    __m256 vec_prod = _mm256_mul_ps(vec_a, vec_b);
                    vec_c = _mm256_add_ps(vec_c, vec_prod);
                    _mm256_storeu_ps(&result.data[i * other.cols + j], vec_c);

                }

                for (; j < other.cols; j++) {
                    result(i, j) += a_val * other(k, j);
                }

            }   
        }
    

    #else 
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t k = 0; k < this->cols; k++) {
                float a_val = (*this)(i, k);
                for (size_t j = 0; j < other.cols; j++) {
                    result(i, j) += a_val * other(k, j);
                }
            }
        }
    #endif 
    
    return result;
}

Matrix Matrix::T() const {
    Matrix result(cols, rows);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (other.rows != this->rows || other.cols != this->cols) {
        throw std::invalid_argument("Shape mismatch");
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < this->data.size(); i++) {
            result.data[i] = this->data[i] - other.data[i];
    }

    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {

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

Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows, cols);

    for (size_t i = 0; i < this->data.size(); i++) {
        result.data[i] = scalar * this->data[i];
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Shape mismathc");
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < this->data.size(); i++) {
        result.data[i] = this->data[i] * other.data[i];
    }

    return result;

}
Matrix Matrix::sum_axis1() const {
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