#include "utils.hpp"
#include "vector.hpp"
#include "matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

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

Vector data_to_vector(const std::string& path) {
    std::ifstream file(path);

    size_t size;
    if (file.is_open())
        file >> size;

    if (size == 0)
        throw std::invalid_argument("size of vector cannot be less than one");
    
    Vector data(size);
    size_t counter{};
    while ((file >> data[counter]) && (++counter != size)) {}
    
    if (counter != size)
        throw std::logic_error("not enough arguments for vector");

    file.close();
    return data;
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

Logger::Logger(const std::string& filename) {
    file_.open(filename, std::ios::app);
    if (!file_.is_open()) {
        std::cerr << "[Ошибка] Не удалось открыть файл лога: " << filename << "\n";
    }
}

Logger::~Logger() { flush(); }

void Logger::log(const std::string& message) {
    if (file_.is_open()) {
        file_ << message << "\n";
    }
}

void Logger::flush() {
    if (file_.is_open()) {
        file_.flush();
    }
}

NetworkLogger::NetworkLogger(const std::string& filename) : Logger(filename) {}

void NetworkLogger::log_vector(const Vector& vec) {  // УБРАЛ name
    if (!file_.is_open()) return;

    file_ << std::fixed << std::setprecision(6);
    file_ << vec.size() << "\n";  // Сначала размерность
    for (size_t i = 0; i < vec.size(); ++i) {
        file_ << vec[i];
        if (i != vec.size() - 1) file_ << " ";
    }
    file_ << "\n";
}

void NetworkLogger::log_matrix(const Matrix& mat) {  // УБРАЛ name
    if (!file_.is_open()) return;

    file_ << std::fixed << std::setprecision(6);
    file_ << mat.rows() << " " << mat.cols() << "\n";
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            file_ << mat[i][j];
            if (j != mat.cols() - 1) file_ << " ";
        }
        file_ << "\n";
    }
}