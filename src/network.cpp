#include "network.hpp"
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iostream>

Network::Network(const WeightVendor& wv, const ActivationVendor& av, DataVendor* dv) {
    n_layers = wv.count();
    this->dv = dv;

    layers = new Layer[n_layers];
    weights = new Weight[n_layers - 1];
    grads = new Weight[n_layers - 1];

    for (size_t i{}; i < n_layers - 1; ++i) {
        const Weight& fetched_weight = wv.fetch(i);
        Weight& weight = weights[i];

        const Matrix& fetched_weight_matrix = fetched_weight.w();
        Matrix& weight_matrix = weight.w();
        weight_matrix = fetched_weight_matrix;

        const Vector& fetched_bias = fetched_weight.b();
        Vector& bias = weight.b();
        bias = fetched_bias;
    }

    for (size_t i{}; i < n_layers; ++i)
        layers[i].activation() = av.fetch(i);
}

Network::~Network() {
    delete[] layers;
    delete[] weights;
    delete[] grads;
}

void Network::vend_data(size_t idx) {
    const Data& test = dv->fetch(idx);
    layers[0].z() = test.first;
    y = test.second;
}

void Network::propagate() {
    for (size_t i{}; i < n_layers - 1; ++i) {
        const Weight& weight = weights[i];
        Vector z = weight.w() * layers[i].az();
        const Vector& b = weight.b();
        if (wb)
            z += weight.b();
        layers[i + 1].z() = std::move(z);
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

    grads[k - 1].w() = outer_product(dl_dz, L_p.az());
    if (wb)
        grads[k - 1].b() = dl_dz;

    for (k = n_layers - 2; k > 0; --k) {
        Layer& L = layers[k];
        Layer& L_p = layers[k - 1];

        dl_dL = dl_dz * weights[k].w();
        dL_dz = L.gz();
        dl_dz = hadamar(dl_dL, dL_dz);

        grads[k - 1].w() = outer_product(dl_dz, L_p.az());
        if (wb)
            grads[k - 1].b() = dl_dz;
    }
}

void Network::apply_grads() {
    FtoF func = [&](float x) { return lr * x; };
    for (size_t i{}; i < n_layers - 1; ++i) {
        weights[i].w() -= grads[i].w().map(func);
        if (wb)
            weights[i].b() -= grads[i].b().map(func);
    }
}

void Network::epochs(size_t epochs) {
    for (size_t i{}; i < epochs; ++i)
        for (size_t j{}; j < dv->count(); ++j) {
            vend_data(j);
            propagate();
            backpropagate();
            apply_grads();
        }
}

Vector Network::compute(const Vector& v) {
    if (v.size() != layers[0].z().size())
        throw std::invalid_argument("Dimentionally inconsistent vector was provided for computation");
    layers[0].z() = v;
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

Network& Network::set_wb(bool wb) {
    this->wb = wb;
    return *this;
}

Network& Network::set_dv(DataVendor* dv) {
    this->dv = dv;
    return *this;
}

Network& Network::set_layer_activation(size_t idx, const Activation& activation) {
    get_layer(idx).activation() = activation;
    return *this;
}

Layer& Network::get_layer(size_t idx) {
    if (idx >= n_layers)
        throw std::out_of_range("Layer index out of range");
    return layers[idx];
}

Weight& Network::get_weight(size_t idx) {
    if (idx >= n_layers - 1)
        throw std::out_of_range("Weight index out of range");
    return weights[idx];
}
