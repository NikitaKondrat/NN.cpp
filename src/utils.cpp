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
        throw std::invalid_argument("same-dimantional vectors required to compute loss partial derivative");
    size_t n = est.size();
    Vector result(n);
    for (size_t i{}; i < n; ++i) 
        result[i] = 2 * (est[i] - ans[i]) / n;
    return result;
}

Vector bce_lp(const Vector& est, const Vector& ans) {
    if (est.size() != ans.size())
        throw std::invalid_argument("same-dimantional vectors required to compute loss partial derivative");
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
        throw std::invalid_argument("same-dimantional vectors required to compute loss partial derivative");
    size_t n = est.size();
    Vector result(n);
    for (size_t i{}; i < n; ++i) {
        result[i] = (est[i] - ans[i]) / n;
    }
    return result;
}

FtoF random_uniform_filler(float a, float b) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(a, b);
    return [dist](float) mutable -> float { return dist(gen); };
}

DataVendor::~DataVendor() {
    delete[] data;
    data = nullptr;
}

const Data& DataVendor::fetch(size_t idx) const {
    return data[idx];
}

size_t DataVendor::ds_size() const {
    return ds_size_;
}

size_t DataVendor::in_size() const {
    return in_size_;
}

size_t DataVendor::out_size() const {
    return out_size_;
}

FileDataVendor::FileDataVendor(const std::string& path) {
    std::ifstream ifs(path);
    ifs >> ds_size_ >> in_size_ >> out_size_;
    data = new Data[ds_size_];
    Vector in(in_size_);
    Vector out(out_size_);
    for (size_t i{}; i < ds_size_; ++i) {
        for (size_t j{}; j < in_size_; ++j)
            ifs >> in.data()[j];
        for (size_t j{}; j < out_size_; ++j)
            ifs >> out.data()[j];
        data[i] = {in, out};
    }
}

ObjectDataVendor::ObjectDataVendor(std::initializer_list<Data> l) {
    ds_size_ = l.size();
    in_size_ = (*l.begin()).first.size();
    out_size_ = (*l.begin()).second.size();
    data = new Data[ds_size_];
    std::copy(l.begin(), l.end(), data);
}

Activation::Activation() : a(id), ad(id_deriv) { }

Activation::Activation(const FtoF& activation, const FtoF& activation_deriv) : a(activation), ad(activation_deriv) { }

Activation::Activation(const Activation& activation) : a(activation.a), ad(activation.ad) { }

Layer::Layer() : z_(Vector()), activation_(Activation()) { }

void Layer::set_z(const Vector& z) {
    z_ = z;
}

void Layer::set_z(Vector&& z) {
    z_ = std::move(z);
}

Activation& Layer::activation() {
    return activation_;
}

const Vector& Layer::z() const {
    return z_;
}

Vector Layer::az() const {
    return z_.map(activation_.a);
}

Vector Layer::gz() const {
    return z_.map(activation_.ad);
}

Weight::Weight() : w_(Matrix()), b_(Vector()) { }

void Weight::set_w(const Matrix& w) {
    w_ = w;
}

void Weight::set_w(Matrix&& w) {
    w_ = std::move(w);
}

void Weight::set_b(const Vector& b) {
    b_ = b;
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
