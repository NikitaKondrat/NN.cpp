#include <iostream>
#include <string>
#include "network.hpp"

void train_bool_op(const std::string& s, DataVendor* dv) {
    size_t n_layers = 4;
    size_t l_size = 3;
    size_t in_size = 2;
    size_t out_size = 1;
    size_t epochs = 100'000;

    Network nw(n_layers, l_size, in_size, out_size);
    nw.fill_weights(random_uniform_filler(-1.0f, 1.0f)).set_dv(dv);

    Activation activation(sigmoid, sigmoid_deriv);
    for (size_t i{1}; i < n_layers; ++i) 
        nw.set_layer_activation(i, activation);

    nw.epochs(epochs);

    Vector est(dv->ds_size());
    for (size_t i{}; i < dv->ds_size(); ++i)
        est[i] = nw.compute(dv->fetch(i).first)[0];

    for (int i = 0; i <= 1; ++i)
        for (int j = 0; j <= 1; ++j)
            std::cout << i << " " << s << " " << j << " = " << est[2 * i + j] << std::endl;
    std::cout << std::endl;
}

int main() {
    ObjectDataVendor xor_ds = {
        {Vector{0, 0}, Vector{0}},
        {Vector{0, 1}, Vector{1}},
        {Vector{1, 0}, Vector{1}},
        {Vector{1, 1}, Vector{0}},
    };
    ObjectDataVendor and_ds = {
        {Vector{0, 0}, Vector{0}},
        {Vector{0, 1}, Vector{0}},
        {Vector{1, 0}, Vector{0}},
        {Vector{1, 1}, Vector{1}},
    };
    ObjectDataVendor or_ds = {
        {Vector{0, 0}, Vector{0}},
        {Vector{0, 1}, Vector{1}},
        {Vector{1, 0}, Vector{1}},
        {Vector{1, 1}, Vector{1}},
    };

    train_bool_op("^", &xor_ds);
    train_bool_op("&", &and_ds);
    train_bool_op("|", &or_ds);
}