#pragma once

#include "vector.hpp"
#include "matrix.hpp"
#include <functional>
#include <string>
#include <random>
#include <vector>

using Data = std::vector<std::pair<Vector, Vector>>;

float id(float);
float id_deriv(float);

float sigmoid(float);
float sigmoid_deriv(float);

Vector mse_lp(const Vector&, const Vector&);

FtoF random_uniform_filler(float, float);

Data parse_data(const std::string&);

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
    Activation activation_;
public:
    Layer();
    void set_z(const Vector&);
    void set_z(Vector&&);
    Activation& activation();
    const Vector& z() const;
    Vector az() const;
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
    Matrix& w();
    Vector& b();
};