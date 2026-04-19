#pragma once

#include <cstddef>
#include <string>
#include "utils.hpp"

class NetworkLogger;

using Loss = std::function<Vector(const Vector&, const Vector&)>;

class Network {
private:
    size_t n_layers;
    Layer* layers;
    Weight* weights;
    Weight* grads;
    Loss lp;
    Vector out;
public:
    Network(size_t, size_t, size_t, size_t);
    ~Network();
    void fill_from_path(std::string);
    void propagate(bool);
    void backpropagate(bool with_out_activ, double lr, NetworkLogger& logger);
    void set_lp(const Loss&);
    Layer& get_layer(size_t idx);
    Weight& get_weight(size_t idx);
};