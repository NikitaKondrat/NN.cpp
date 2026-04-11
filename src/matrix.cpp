#include "matrix.hpp"
#include <stdexcept>
#include <algorithm>

Matrix::Matrix(size_t n_rows_, size_t n_cols_, Row* matrix_) : n_rows(n_rows_), n_cols(n_cols_), matrix(matrix_ ? matrix_ : new Row[n_rows_]) { }

Matrix::Matrix() : Matrix(0, 0, nullptr) { }

Matrix::Matrix(size_t n_rows, size_t n_cols) : Matrix(n_rows, n_cols, nullptr) {
    for (size_t i{}; i < n_rows; ++i)
        matrix[i] = Row(n_cols);
}

Matrix::Matrix(std::initializer_list<Row> l) : Matrix(l.size(), l.size() > 0 ? l.begin()->size() : 0, nullptr) {
    std::copy(l.begin(), l.end(), matrix);
}

Matrix::Matrix(const Matrix& other) : Matrix(other.n_rows, other.n_cols, nullptr) {
    std::copy(other.matrix, other.matrix + other.n_rows, matrix);
}

Matrix::Matrix(Matrix&& other) noexcept {
    swap(other);
    other.n_rows = 0;
    other.n_cols = 0;
    other.matrix = nullptr;
}

Matrix::~Matrix() {
    delete[] matrix;
}

Matrix& Matrix::operator=(Matrix other) {
    swap(other);
    return *this;
}

const Row& Matrix::operator[](size_t idx) const {
    if (idx >= n_rows)
        throw std::out_of_range("row index out of range");
    return matrix[idx];
}

Row& Matrix::operator[](size_t idx) {
    return const_cast<Row&>(static_cast<const Matrix&>(*this)[idx]);
}

Matrix Matrix::operator-(const Matrix& other) const {
    return apply_op(other, op::sub);
}

Matrix& Matrix::operator-=(const Matrix& other) {
    Matrix result = apply_op(other, op::sub);
    swap(result);
    return *this;
}

Vector Matrix::operator*(const Vector& v) const {
    if (v.size() != n_cols)
        throw std::invalid_argument("dimentional inconsistency in matrix-vector multiplication");
    Vector result(n_rows);
    float* result_values = result.data();
    const float* v_values = v.data();
    for (size_t i{}; i < n_rows; ++i) {
        const float* row = matrix[i].data();
        float result_i{};
        for (size_t j{}; j < n_cols; ++j)
            result_i += row[j] * v_values[j];
        result_values[i] = result_i;
    }
    return result;
}

Matrix& Matrix::apply(const FtoF& func) {
    for (size_t i{}; i < n_rows; ++i) {
        float* row = matrix[i].data();
        for (size_t j{}; j < n_cols; ++j)
            row[j] = func(row[j]);
    }
    return *this;
}

size_t Matrix::rows() const {
    return n_rows;
}

size_t Matrix::cols() const {
    return n_cols;
}

Row* Matrix::data() {
    return matrix;
}

const Row* Matrix::data() const {
    return matrix;
}

void Matrix::swap(Matrix& other) noexcept {
    std::swap(n_rows, other.n_rows);
    std::swap(n_cols, other.n_cols);
    std::swap(matrix, other.matrix);
}

Matrix Matrix::apply_op(const Matrix& other, const FFtoF& op) const {
    if (n_rows != other.n_rows || n_cols != other.n_cols)
        throw std::invalid_argument("dimentional inconsistency in applied matrix operation");
    Matrix result = Matrix(n_rows, n_cols);
    for (size_t i{}; i < n_rows; ++i) {
        const float* this_row = matrix[i].data();
        const float* other_row = other.matrix[i].data();
        float* result_row = result.matrix[i].data();
        for (size_t j{}; j < n_cols; ++j)
            result_row[j] = op(this_row[j], other_row[j]);
    }
    return result;
}

Matrix operator*(float a, const Matrix& matrix) {
    Matrix result(matrix.n_rows, matrix.n_cols);
    for (size_t i{}; i < matrix.n_rows; ++i) {
        const float* matrix_row = matrix.data()[i].data();
        float* result_row = result.data()[i].data();
        for (size_t j{}; j < matrix.n_cols; ++j)
            result_row[j] = a * matrix_row[j];
    }
    return result;
}

Matrix diag(const Vector& v) {
    Matrix matrix(v.size(), v.size());
    const float* v_values = v.data();
    for (size_t i{}; i < v.size(); ++i) {
        float* row = matrix.data()[i].data();
        row[i] = v_values[i];
    }
    return matrix;
}

Matrix outer_product(const Vector& u, const Vector& v) {
    Matrix matrix(u.size(), v.size());
    const float* u_values = u.data();
    const float* v_values = v.data();
    for (size_t i{}; i < u.size(); ++i) {
        float* row = matrix.data()[i].data();
        const float u_value = u_values[i];
        for (size_t j{}; j < v.size(); ++j)
            row[j] = u_value * v_values[j];
    }
    return matrix;
}

Vector operator*(const Vector& v, const Matrix& matrix) {
    if (v.size() != matrix.rows())
        throw std::invalid_argument("dimentional inconsistency in vector-matrix multiplication");
    Vector result(matrix.cols());
    const float* v_values = v.data();
    float* result_values = result.data();
    for (size_t i{}; i < matrix.rows(); ++i) {
        const float* matrix_row = matrix.data()[i].data();
        const float v_value = v_values[i];
        for (size_t j{}; j < matrix.cols(); ++j)
            result_values[j] += v_value * matrix_row[j];
    }
    return result;
}