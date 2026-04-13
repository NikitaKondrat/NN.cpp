#pragma once

#include "vector.hpp"
#include "matrix.hpp"
#include <functional>
#include <string>
#include <fstream>

float id(float);
float id_deriv(float);

float sigmoid(float);
float sigmoid_deriv(float);

Vector mse_lp(const Vector&, const Vector&);

Vector data_to_vector(const std::string&);

class Activation {
public:
    FtoF activ;
    FtoF activ_deriv;
    Activation();
    Activation(const FtoF&, const FtoF&);
    Activation(const Activation&);
};

class Layer {
private:
    Vector z_;
    Activation activ_;
public:
    Layer();
    void set_z(const Vector&);
    void set_z(Vector&&);
    Activation& activ();
    const Vector& z() const;
    Vector gz() const;
};

class Weight {
private:
    Matrix w_;
    Vector b_;
public:
    Weight();
    void set_w(const Matrix&);
    void set_b(const Vector&);
    void set_w(Matrix&&);
    void set_b(Vector&&);
    const Matrix& w() const;
    const Vector& b() const;
};

class Logger {
private:
    std::ofstream file_;
public:
    explicit Logger(const std::string& filename = "training.txt");
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void log(const std::string& message);
    void flush();
};

class NetworkLogger : public Logger {
public:
    explicit NetworkLogger(const std::string& filename = "network_data.txt");

    void log_vector(const std::string& name, const Vector& vec);
    void log_matrix(const std::string& name, const Matrix& mat);
};