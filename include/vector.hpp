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
};