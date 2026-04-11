#include "utils.hpp"
#include <cmath>
#include <stdexcept>
#include <fstream>

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

Data parse_data(const std::string& path) {
    std::ifstream file(path);
    size_t dataset_size;
    size_t test_size;
    size_t ans_size;
    if (file.is_open()) {
        file >> dataset_size;
        file >> test_size;
        file >> ans_size;
    } else
        throw std::runtime_error("couldn't open file");

    Data dataset(dataset_size);
    Vector test_data(test_size);
    Vector test_ans(ans_size);
    for (size_t i{}; i < dataset_size; ++i) {
        for (size_t j{}; j < test_size; ++j)
            file >> test_data[j];
        for (size_t j{}; j < ans_size; ++j)
            file >> test_ans[j];
        dataset[i] = {test_data, test_ans};
    }

    file.close();
    return dataset;
}

FtoF random_uniform_filler(float a, float b) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(a, b);
    return [dist](float) mutable -> float { return dist(gen); };
}

Activation::Activation() : activ(id), activ_deriv(id_deriv) { }

Activation::Activation(const FtoF& activ, const FtoF& activ_deriv) : activ(activ), activ_deriv(activ_deriv) { }

Activation::Activation(const Activation& activ) : activ(activ.activ), activ_deriv(activ.activ_deriv) { }

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
    return z_.map(activation_.activ);
}

Vector Layer::gz() const {
    return z_.map(activation_.activ_deriv);
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

Matrix& Weight::w() {
    return w_;
}

Vector& Weight::b() {
    return b_;
}
