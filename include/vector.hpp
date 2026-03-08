#pragma once

#include <cstddef>

class Vector {
private:
    size_t n;
    float* values = nullptr;
public:
    Vector() = default;
    Vector(size_t);
    ~Vector();
    float& operator[](size_t);
    Vector(const Vector& other);
    Vector& operator=(const Vector& other);
};