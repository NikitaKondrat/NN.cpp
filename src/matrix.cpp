#include "matrix.hpp"
#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols)
    : n_rows(rows), n_cols(cols) {

    if (rows > 0 && cols > 0) {
        matrix = new Row[rows];
        for (size_t i = 0; i < rows; ++i) {
            matrix[i] = Vector(cols);
        }
    }
}

Matrix::~Matrix() {
    delete[] matrix;
    matrix = nullptr;
}

Row& Matrix::operator[](size_t index) {
    if (index >= n_rows) {
        throw std::out_of_range("Matrix row index out of range");
    }
    return matrix[index];
}

// Умножение матрицы на вектор
Vector Matrix::operator*(const Vector& vec) {
    if (n_cols != vec.n) {
        throw std::invalid_argument(
            "Matrix columns (" + std::to_string(n_cols) +
            ") must match vector size (" + std::to_string(vec.n) + ")"
        );
    }

    Vector result(n_rows);

    for (size_t i = 0; i < n_rows; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < n_cols; ++j) {
            sum += matrix[i][j] * vec[j];
        }
        result[i] = sum;
    }

    return result;
}