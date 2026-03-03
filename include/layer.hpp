#pragma once

#include <cstddef>

class Layer {
private:
    size_t n;
    float* neurons;
public:
    Layer(size_t);
    ~Layer();
    float& operator[](size_t);
};