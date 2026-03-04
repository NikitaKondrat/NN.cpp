#pragma once

#include <cstddef>
#include <vector>
#include "vector.hpp"
#include "matrix.hpp"

using Weight = Matrix;
using Layer = Vector;

class Network {
private:
    Weight* weights;
    Layer* layers;
public:
    Network(size_t, size_t, size_t); // num. of layers, dim. of input layer, dim. of output layer
    ~Network();
    void get_data(std::vector<float>); // for input layer; primitive version
    void propagate(); // forward-propagation
};