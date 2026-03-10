#pragma once

#include "vector.hpp"
#include "matrix.hpp"
#include <functional>

float id(float);
float id_deriv(float);

float sigmoid(float);
float sigmoid_deriv(float);

Vector mse_lp(const Vector&, const Vector&);

class Activation {
public:
    float (*activ)(float);
    float (*activ_deriv)(float);
    Activation();
    Activation(float (*)(float), float (*)(float));
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