#include "network.hpp"
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iostream>

Network::Network(
    size_t n_layers, size_t l_size, 
    size_t in_size, size_t out_size
) {
    this->n_layers = n_layers;
    
    grads = new Weight[n_layers - 1];

    layers = new Layer[n_layers];
    for (size_t i{1}; i < n_layers - 1; ++i)
        layers[i].set_z(Vector(l_size));
    layers[0].set_z(Vector(in_size));
    layers[n_layers - 1].set_z(Vector(out_size));

    weights = new Weight[n_layers - 1];
    for (size_t i{1}; i < n_layers - 2; ++i)
        weights[i].set_w(Matrix(l_size, l_size));
    weights[0].set_w(Matrix(l_size, in_size));
    weights[n_layers - 2].set_w(Matrix(out_size, l_size));

    fill_weights(random_uniform_filler(-1.0f, 1.0f));
}

Network::~Network() {
    delete[] layers;
    delete[] weights;
    delete[] grads;
}

void Network::fill_with_path(const std::string& path) {
    data = parse_data(path);
}

void Network::fill_with_dataset(const Data& dataset) {
    data = dataset;
}

void Network::fill_weights(const FtoF& func) {
    for (size_t i{}; i < n_layers - 1; ++i) 
        weights[i].w().apply(func);
}

void Network::propagate() {
    for (size_t i{}; i < n_layers - 1; ++i) {
        const Weight& weight = weights[i];
        Vector z = weight.w() * layers[i].az();
        const Vector& b = weight.b();
        if (with_bias)
            z += weight.b();
        layers[i + 1].set_z(z);
    }
}

void Network::backpropagate() {
    size_t k = n_layers - 1;
    Layer& L = layers[k];
    Layer& L_p = layers[k - 1];

    Vector dl_dL, dL_dz, dl_dz;
    dl_dL = lp(L.az(), y);
    dL_dz = L.gz();
    dl_dz = hadamar(dl_dL, dL_dz);

    grads[k - 1].set_w(outer_product(dl_dz, L_p.az()));
    if (with_bias)
        grads[k - 1].set_b(dl_dz);

    for (k = n_layers - 2; k > 0; --k) {
        Layer& L = layers[k];
        Layer& L_p = layers[k - 1];

        dl_dL = dl_dz * weights[k].w();
        dL_dz = L.gz();
        dl_dz = hadamar(dl_dL, dL_dz);

        grads[k - 1].set_w(outer_product(dl_dz, L_p.az()));
        if (with_bias)
            grads[k - 1].set_b(dl_dz);
    }
}

void Network::apply_grads(float rate) {
    for (size_t i{}; i < n_layers - 1; ++i) {
        weights[i].w() -= rate * grads[i].w();
        if (with_bias)
            weights[i].b() -= rate * grads[i].b();
    }
}

void Network::epochs(size_t epochs) {
    for (size_t i{}; i < epochs; ++i) 
        for (const auto& test : data) {
            layers[0].set_z(test.first);
            y = test.second;
            propagate();
            backpropagate();
            apply_grads(lr);
        }
}

Vector Network::compute(const Vector& v) {
    layers[0].set_z(v);
    propagate();
    return layers[n_layers - 1].az();
}

Network& Network::set_lp(const Loss& lp) {
    this->lp = lp;
    return *this;
}

Network& Network::set_lr(float lr) {
    this->lr = lr;
    return *this;
}

Network& Network::set_with_bias(bool with_bias) {
    this->with_bias = with_bias;
    return *this;
}

Network& Network::set_layer_activation(size_t idx, const Activation& activation) {
    get_layer(idx).activation() = activation;
    return *this;
}

Layer& Network::get_layer(size_t idx) {
    if (idx >= n_layers)
        throw std::out_of_range("layer index out of range");
    return layers[idx];
}

Weight& Network::get_weight(size_t idx) {
    if (idx >= n_layers - 1)
        throw std::out_of_range("weight index out of range");
    return weights[idx];
}
