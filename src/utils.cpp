#include "utils.hpp"
#include <cmath>
#include <stdexcept>

float id(float x) {
    return x;
}

float id_deriv(float x) {
    return 1.0f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_deriv(float x) {
    float xp = std::exp(-x);
    return xp / ((1 + xp) * (1 + xp));
}

Vector mse_lp(const Vector& est, const Vector& corr) {
    if (est.size() != corr.size())
        throw std::invalid_argument("same-dimantional vectors required to compute loss partial derivative");
    Vector result(est.size());
    for (size_t i{}; i < est.size(); ++i) 
        result[i] = 2 * (est[i] - corr[i]);
    return result;
}

Activation::Activation() : activ(id), activ_deriv(id_deriv) { }

Activation::Activation(const FtoF& activ, const FtoF& activ_deriv) : activ(activ), activ_deriv(activ_deriv) { }

Activation::Activation(const Activation& activ) : activ(activ.activ), activ_deriv(activ.activ_deriv) { }

Layer::Layer() : z_(Vector()), activ_(Activation()) { }

void Layer::set_z(const Vector& z) {
    z_ = z;
}

void Layer::set_z(Vector&& z) {
    z_ = std::move(z);
}

Activation& Layer::activ() {
    return activ_;
}

const Vector& Layer::z() const {
    return z_;
}

Vector Layer::gz() const {
    return z_.map(activ_.activ_deriv);
}

Weight::Weight() : w_(Matrix()), b_(Vector()) { }

void Weight::set_w(const Matrix& w) {
    w_ = w;
}

void Weight::set_b(const Vector& b) {
    b_ = b;
}

void Weight::set_w(Matrix&& w) {
    w_ = std::move(w);
}

void Weight::set_b(Vector&& b) {
    b_ = std::move(b);
}

const Matrix& Weight::w() const {
    return w_;
}

const Vector& Weight::b() const {
    return b_;
}
