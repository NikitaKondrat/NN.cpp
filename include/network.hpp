#pragma once

#include <cstddef>
#include <string>
#include <random>
#include "utils.hpp"

using Loss = std::function<Vector(const Vector&, const Vector&)>;

class Network {
private:
    size_t n_layers;
    Layer* layers;
    Weight* weights;
    Weight* grads;
    Vector y;
    Loss lp = mse_lp;
    float lr = 0.1;
    bool with_bias = false;
    Data data;

public:
    Network(size_t, size_t, size_t, size_t);
    ~Network();
    void fill_with_path(const std::string&);
    void fill_with_dataset(const Data&);
    void fill_weights(const FtoF&);
    void propagate();
    void backpropagate();
    void apply_grads(float);
    void epochs(size_t);
    Vector compute(const Vector&);
    Network& set_lp(const Loss&);
    Network& set_lr(float);
    Network& set_with_bias(bool);
    Network& set_layer_activation(size_t idx, const Activation&);
    Layer& get_layer(size_t idx);
    Weight& get_weight(size_t idx);
};