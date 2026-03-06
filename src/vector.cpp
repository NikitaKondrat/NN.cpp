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