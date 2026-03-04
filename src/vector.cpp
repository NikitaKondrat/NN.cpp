#include "vector.hpp"
#include <stdexcept>

Vector::Vector(size_t n) {
    this->n = n;
    values = new float[n]{};
}

Vector::~Vector() {
    delete[] values;
}

float& Vector::operator[](size_t idx) {
    if (idx > n)
        throw std::out_of_range("layer index out of range");
    return values[idx];
}