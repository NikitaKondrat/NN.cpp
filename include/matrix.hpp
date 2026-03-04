#pragma once

#include <cstddef>
#include "vector.hpp"

using Row = Vector;

class Matrix {
private:
    size_t n_rows, n_cols;
    Row* matrix = nullptr;
public:
    Matrix() = default;
    Matrix(size_t, size_t); // num. of rows and num. of columns
    ~Matrix();
    Row& operator[](size_t);
    Vector operator*(const Vector&);
};