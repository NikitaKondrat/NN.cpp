#include "network.hpp"
#include <stdexcept>
#include <fstream>
#include "utils.hpp"

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

    for (size_t i = 0; i < n_layers - 1; ++i) {
        if (i == 0) {
            grads[i].set_w(Matrix(l_size, in_size));
            grads[i].set_b(Vector(l_size));
        }
        else if (i == n_layers - 2) {
            grads[i].set_w(Matrix(out_size, l_size));
            grads[i].set_b(Vector(out_size));
        }
        else {
            grads[i].set_w(Matrix(l_size, l_size));
            grads[i].set_b(Vector(l_size));
        }
    }

}

Network::~Network() {
    delete[] layers;
    delete[] weights;
    delete[] grads;
}

void Network::fill_from_path(std::string path) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("couldn't open file");

    size_t in_size = layers[0].z().size();
    Vector in_data(in_size);
    for (size_t i = 0; i < in_size; ++i)
        in >> in_data[i];
    layers[0].set_z(std::move(in_data));

    size_t out_size = layers[n_layers - 1].z().size();
    for (size_t i = 0; i < out_size; ++i)
        in >> out[i];

    in.close();
}

void Network::propagate(bool with_bias = false) {
    for (size_t i{}; i < n_layers - 1; ++i) {
        const Weight& weight = weights[i];
        Vector z = weight.w() * layers[i].gz();
        const Vector& b = weight.b();
        if (with_bias)
            z += weight.b();
        layers[i + 1].set_z(z);
    }
}

void Network::backpropagate(bool with_out_activ = true) {
    size_t k = n_layers - 1;
    Layer& L = layers[k];
    Layer& L_p = layers[k - 1];

    Vector dl_dL, dL_dz, dl_dz;
    if (with_out_activ) {
        dl_dL = lp(L.gz(), out);
        dL_dz = L.z().map(L.activ().activ_deriv);
        dl_dz = hadamar(dl_dL, dL_dz);
    } else
        dl_dz = lp(L.gz(), out);

    grads[k - 1].set_w(outer_product(dl_dz, L_p.gz()));
    grads[k - 1].set_b(dl_dz);

    for (k = n_layers - 2; k > 0; --k) {
        L = layers[k];
        L_p = layers[k - 1];

        dl_dL = dl_dz * weights[k].w();
        dL_dz = L.z().map(L.activ().activ_deriv);
        dl_dz = hadamar(dl_dL, dL_dz);

        grads[k - 1].set_w(outer_product(dl_dz, L_p.gz()));
        grads[k - 1].set_b(dl_dz);
    }
}

void Network::set_lp(const Loss& lp) {
    this->lp = lp;
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

void Network::update_weights(double lr, NetworkLogger& logger) {
    logger.log("UPDATE");

    for (size_t i = 0; i < n_layers - 1; ++i) {
        const Matrix& grad_w = grads[i].w();
        const Vector& grad_b = grads[i].b();

        const Matrix& current_w = weights[i].w();
        const Vector& current_b = weights[i].b();

        Matrix new_w(current_w.rows(), current_w.cols());
        for (size_t r = 0; r < current_w.rows(); ++r)
            for (size_t c = 0; c < current_w.cols(); ++c)
                new_w[r][c] = current_w[r][c] - lr * grad_w[r][c];

        Vector new_b(current_b.size());
        for (size_t j = 0; j < current_b.size(); ++j)
            new_b[j] = current_b[j] - lr * grad_b[j];

        weights[i].set_w(std::move(new_w));
        weights[i].set_b(std::move(new_b));

        logger.log_matrix(weights[i].w());
        logger.log_vector(weights[i].b());
    }
}