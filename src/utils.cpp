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

DataVendor::~DataVendor() {
    delete[] data;
    data = nullptr;
}

const Data& DataVendor::fetch(size_t idx) {
    return data[idx];
}

size_t DataVendor::in_size() const {
    return in_size_;
}

size_t DataVendor::out_size() const {
    return out_size_;
}

FileDataVendor::FileDataVendor(const std::string& path) {
    std::ifstream ifs(path);
    ifs >> count_ >> in_size_ >> out_size_;
    data = new Data[count_];
    for (size_t i{}; i < count_; ++i) {
        Vector in(in_size_);
        Vector out(out_size_);
        float* in_data = in.data();
        float* out_data = out.data();
        for (size_t j{}; j < in_size_; ++j)
            ifs >> in_data[j];
        for (size_t j{}; j < out_size_; ++j)
            ifs >> out_data[j];
        data[i] = {std::move(in), std::move(out)};
    }
}

ObjectDataVendor::ObjectDataVendor(std::initializer_list<Data> l) {
    count_ = l.size();
    in_size_ = l.begin()->first.size();
    out_size_ = l.begin()->second.size();
    data = new Data[count_];
    std::copy(l.begin(), l.end(), data);
}

WeightVendor::~WeightVendor() {
    delete[] weights;
    weights = nullptr;
}

const Weight& WeightVendor::fetch(size_t idx) {
    return weights[idx];
}

FileWeightVendor::FileWeightVendor(const std::string& path) {
    std::ifstream ifs(path);
    ifs >> count_ >> with_bias_;
    weights = new Weight[count_];

    for (size_t l{}; l < count_; ++l) {
        size_t r, c;
        ifs >> r >> c;
        Matrix weight(r, c);
        for (size_t i{}; i < r; ++i) {
            float* row_data = weight[i].data();
            for (size_t j{}; j < c; ++j)
                ifs >> row_data[j];
        }
        weights[l].set_w(std::move(weight));

        if (with_bias_) {
            Vector bias(r);
            float* bias_data = bias.data();
            for (size_t i{}; i < r; ++i)
                ifs >> bias_data[i];
            weights[l].set_b(std::move(bias));
        }
    }
}

ObjectWeightVendor::ObjectWeightVendor(std::initializer_list<Matrix> l) {
    count_ = l.size();
    with_bias_ = false;
    weights = new Weight[count_];
    size_t i{};
    for (const auto& weight : l)
        weights[i].set_w(weight);
}

ObjectWeightVendor::ObjectWeightVendor(std::initializer_list<std::pair<Matrix, Vector>> l) {
    count_ = l.size();
    with_bias_ = true;
    weights = new Weight[count_];
    size_t i{};
    for (const auto& weight: l) {
        weights[i].set_w(weight.first);
        weights[i].set_b(weight.second);
    }
}

RandomWeightVendor::RandomWeightVendor(
    size_t n_layers, size_t l_size, size_t in_size, size_t out_size, 
    bool with_bias, float a, float b
) : gen(std::random_device{}()), dist(a, b) {
    count_ = n_layers;
    with_bias_ = with_bias;
    auto func = [this](float) mutable -> float { return dist(gen); };
    weights = new Weight[n_layers - 1];

    weights[0].set_w(Matrix(l_size, in_size).map(func));
    for (size_t i{1}; i < n_layers - 2; ++i) 
        weights[i].set_w(Matrix(l_size, l_size).map(func));
    weights[n_layers - 2].set_w(Matrix(out_size, l_size).map(func));

    if (with_bias) {
        for (size_t i{}; i < n_layers - 2; ++i)
            weights[i].set_b(Vector(l_size).map(func));
        weights[n_layers - 2].set_b(Vector(out_size).map(func));
    }
}