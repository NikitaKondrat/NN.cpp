#include "utils.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

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
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float tgh(float x) {
    return std::tanh(x);
}

float tgh_deriv(float x) {
    float t = tgh(x);
    return 1.0f - t * t;
}

float relu(float x) {
    return x <= 0.0f ? 0.0f : x;
}

float relu_deriv(float x) {
    return x <= 0.0f ? 0.0f : 1.0f;
}

Vector mse_lp(const Vector& est, const Vector& ans) {
    if (est.size() != ans.size())
        throw std::invalid_argument("Same-dimantional vectors required to compute loss partial derivative");
    size_t n = est.size();
    Vector result(n);
    for (size_t i{}; i < n; ++i) 
        result[i] = 2 * (est[i] - ans[i]) / n;
    return result;
}

Vector bce_lp(const Vector& est, const Vector& ans) {
    if (est.size() != ans.size())
        throw std::invalid_argument("Same-dimantional vectors required to compute loss partial derivative");
    size_t n = est.size();
    Vector result(n);
    float eps = 1e-7;
    for (size_t i{}; i < n; ++i) {
        float yh = std::clamp(est[i], eps, 1.0f - eps);
        result[i] = (yh - ans[i]) / (yh * (1 - yh) * n);
    }
    return result;
}

Vector cce_lp(const Vector& est, const Vector& ans) {
    if (est.size() != ans.size())
        throw std::invalid_argument("Same-dimantional vectors required to compute loss partial derivative");
    size_t n = est.size();
    Vector result(n);
    for (size_t i{}; i < n; ++i) {
        result[i] = (est[i] - ans[i]) / n;
    }
    return result;
}

Activation::Activation() : a(id), ad(id_deriv) { }

Activation::Activation(const FtoF& activation, const FtoF& activation_deriv) : a(activation), ad(activation_deriv) { }

Activation::Activation(const Activation& activation) : a(activation.a), ad(activation.ad) { }

Layer::Layer() : z_(Vector()), activation_(Activation()) { }

Activation& Layer::activation() noexcept {
    return activation_;
}

Vector& Layer::z() noexcept {
    return z_;
}

const Vector& Layer::z() const noexcept {
    return z_;
}

Vector Layer::az() const {
    return z_.map(activation_.a);
}

Vector Layer::gz() const {
    return z_.map(activation_.ad);
}

Weight::Weight() : w_(Matrix()), b_(Vector()) { }

Matrix& Weight::w() noexcept {
    return w_;
}

const Matrix& Weight::w() const noexcept {
    return w_;
}

Vector& Weight::b() noexcept {
    return b_;
}

const Vector& Weight::b() const noexcept {
    return b_;
}