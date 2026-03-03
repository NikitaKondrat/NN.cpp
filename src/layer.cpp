#include "layer.hpp"
#include <stdexcept>

Layer::Layer(size_t n) {
    this->n = n;
    neurons = new float[n]{};
}

Layer::~Layer() {
    delete[] neurons;
}

float& Layer::operator[](size_t idx) {
    if (idx > n)
        throw std::out_of_range("layer index out of range");
    return neurons[idx];
}