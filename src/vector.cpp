#include "vector.hpp"
#include <stdexcept>

Vector::Vector(size_t size) : n(size) {
    if (n > 0) {
        values = new float[n]();
    }
}

Vector::~Vector() {
    delete[] values;
    values = nullptr;
}

float& Vector::operator[](size_t index) {
    if (index >= n) {
        throw std::out_of_range("Vector index out of range");
    }
    return values[index];
}

Vector::Vector(const Vector& other) : n(other.n) {
    if (n > 0) {
        values = new float[n];
        for (size_t i = 0; i < n; ++i) {
            values[i] = other.values[i];
        }
    }
    else {
        values = nullptr;
    }
}

Vector& Vector::operator=(const Vector& other) {
    if (this != &other) {
        delete[] values;
        n = other.n;
        if (n > 0) {
            values = new float[n];
            for (size_t i = 0; i < n; ++i) {
                values[i] = other.values[i];
            }
        }
        else {
            values = nullptr;
        }
    }
    return *this;
}