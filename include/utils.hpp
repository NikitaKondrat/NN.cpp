#pragma once

#include "vector.hpp"
#include "matrix.hpp"
#include <functional>
#include <string>
#include <random>
#include <fstream>
#include <initializer_list>

using Data = std::pair<Vector, Vector>;

/**
 * @defgroup activation_functions Activation Functions
 * @brief Functions and their derivatives for neural network layers.
 * 
 * All functions have the signature `float(float)`.
 * 
 * @param x Input value (real number).
 * @return Value of a function at a given point `x`.
 * @{
 */
float id(float x);
float id_deriv(float x);

float sigmoid(float x);
float sigmoid_deriv(float x);

float tgh(float x);
float tgh_deriv(float x);

float relu(float x);
float relu_deriv(float x);
/** @} */

/**
 * @defgroup loss_derivatives Loss Function Derivatives
 * @brief Partial derivatives of loss functions with respect to the output layer of a neural network.
 * 
 * All functions have signature `Vector(const Vector&, const Vector&)`.
 * 
 * @param est Vector of estimated (predicted) values.
 * @param ans Vector of target (ground truth values) values.
 * @return Vector of partial derivatives computed element-wise.
 * @{
 */
Vector mse_lp(const Vector& est, const Vector& ans);
Vector bce_lp(const Vector& est, const Vector& ans);
Vector cce_lp(const Vector& est, const Vector& ans);
/** @} */

/**
 * @brief Wrapper class for activation function and its derivative.
 *
 * Instances of this class are used to configure activation behavior
 * for layers in a neural network.
 */
class Activation {
public:
    FtoF a;   // activation function
    FtoF ad;  // derivative of an activation function
    Activation();
    Activation(const FtoF& activation, const FtoF& activation_deriv);
    Activation(const Activation& activation);
};

/**
 * @brief Class representing a single layer in a neural network.
 */
class Layer {
private:
    Vector z_;               // values of neurons
    Activation activation_;  // activation function and its derivative
public:
    Layer();

    /**
     * @defgroup layer_getters Layer Accessors
     * @brief Neurons getter.
     * @return Referecnce to the vector of values of the layers neurons.
     * @{
     */
    Vector& z() noexcept;
    const Vector& z() const noexcept;
    /** @} */

    /**
     * @brief Activation function and its derivative getter and setter.
     * @return Reference to the `Activation`.
     */
    Activation& activation() noexcept;

    /**
     * @brief Gets neurons values with applied activation function.
     * @return Vector of activated neurons.
     */
    Vector az() const;

    /**
     * @brief Gets neurons values with applied derivative of the activation.
     * @return Vector of neurons with applied derivative of the activation.
     */
    Vector gz() const;
};

/**
 * @brief Class representing a single weight in a neural network.
 * 
 * Stores weight matrix and bias vector.
 */
class Weight {
private:
    Matrix w_;  // weight matrix
    Vector b_;  // bias vector
public:
    Weight();

    /**
     * @defgroup weights_weight_getters Weight Matrix Weight Accessors
     * @brief Weight matrix getters.
     * @return Weight matrix as a reference or const reference.
     * @{
     */
    Matrix& w() noexcept;
    const Matrix& w() const noexcept;
    /** @} */

    /**
     * @defgroup bias_weight_getters Bias Weight Accessors
     * @brief Bias vector getter.
     * @return Bias vector as a reference or const reference.
     */
    Vector& b() noexcept;
    const Vector& b() const noexcept;
    /** @} */
};