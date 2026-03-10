#pragma once

#include <cstddef>
#include <initializer_list>
#include <functional>
#include "vector.hpp"

using Row = Vector;

class Matrix {
private:
    size_t n_rows, n_cols;
    Row* matrix;
    Matrix(size_t, size_t, Row*);
    void swap(Matrix&) noexcept;
    Matrix apply_op(const Matrix&, float(*)(const float&, const float&)) const;
public:
    Matrix();
    Matrix(size_t, size_t);
    Matrix(std::initializer_list<Row>);
    Matrix(const Matrix&);
    Matrix(Matrix&&) noexcept;
    ~Matrix();
    Matrix& operator=(Matrix);
    const Row& operator[](size_t) const;
    Row& operator[](size_t);
    Matrix operator-(const Matrix&) const;
    Matrix& operator-=(const Matrix&);
    Vector operator*(const Vector&) const;
    size_t rows() const;
    size_t cols() const;
    Row* data();
    const Row* data() const;
};

Matrix diag(const Vector&);
Matrix outer_product(const Vector&, const Vector&);
Vector operator*(const Vector&, const Matrix&);