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
 * @brief Provider of a filler function for weights initialization in a neural network.
 * @note Uses a static Mersenne Twister generator.
 * @param a Lower bound of the distribution.
 * @param b Upper bound of the distribution.
 * @return Filler function.
 */
FtoF random_uniform_filler(float a, float b);

/**
 * @brief Base class for data providers in a neural network.
 * @note This class itself is not used to construct instances of data providers,
 *       use derived classes instead. All restrictions, however, MUST be observed in every derived class.
 * @warning When used with a neural network:
 *          - `in_size_` MUST be equal to the input layer size (`layers[0].z().size()`)
 *          - `out_size_` MUST be equal to the target vector size (`y.size()`)
 */
class DataVendor {
protected:
    size_t ds_size_ = 0;   // dataset size
    size_t in_size_ = 0;   // input vector size
    size_t out_size_ = 0;  // target vector size
    Data* data = nullptr;  // stored data
    DataVendor(const DataVendor&) = delete;
    DataVendor& operator=(const DataVendor&) = delete;  
public:
    DataVendor() = default;
    virtual ~DataVendor();

    /**
     * @brief Fetches a single sample from the dataset.
     * @warning If `idx` >= `ds_size_` it is an UB.
     * @param idx Zero-based index of the data sample to fetch from `data`.
     * @return Constant reference of a `std::pair` instance, where
     *         - first item is input vector data
     *         - second item is target vector data
     */
    const Data& fetch(size_t idx) const;

    /**
     * @brief Dataset size getter.
     * @return Dataset size.
     */
    size_t ds_size() const;

    /**
     * @brief Input vector size getter.
     * @return Input vector size.
     */
    
    size_t in_size() const;

    /**
     * Target vector size getter.
     * @return Target vector size. 
     */
    size_t out_size() const;
};

/**
 * @brief Class of a data provider that loads data from file.
 */
class FileDataVendor : public DataVendor {
public:
    FileDataVendor(const std::string&);
};

/**
 * @brief Class of a data provider that stores data copied from an initializer list.
 */
class ObjectDataVendor : public DataVendor {
public:
    ObjectDataVendor(std::initializer_list<Data> l);
};

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
    Activation(const Activation&);
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
     * @defgroup neurons_setters Neurons Setters
     * @brief Sets neurons values.
     * @param z Values to set.
     */
    void set_z(const Vector& z);
    void set_z(Vector&& z);

    /**
     * @brief Activation function and its derivative getter and setter.
     * @return Reference to the `Activation`.
     */
    Activation& activation();

    /**
     * @brief Neurons getter.
     * @return Vector of values of the layers neurons.
     */
    const Vector& z() const;

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
     * @defgroup weight_setters Weight Setters
     * @brief Weight matrix setters.
     * @param w Weight matrix.
     * @{
     */
    void set_w(const Matrix& w);
    void set_w(Matrix&& w);
    /** @} */

    /**
     * @defgroup bias_setters Bias Setters
     * @brief Bias vector setters.
     * @param b Bias vector.
     * @{
     */
    void set_b(const Vector& b);
    void set_b(Vector&& b);
    /** @} */

    /**
     * @brief Weight matrix getter.
     * @return Weight matrix as a const reference.
     */
    const Matrix& w() const;

    /**
     * @brief Bias vector getter.
     * @return Bias vector as a const reference.
     */
    const Vector& b() const;
};