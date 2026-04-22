#pragma once

#include <cstddef>
#include <string>
#include <random>
#include "utils.hpp"

using Loss = std::function<Vector(const Vector&, const Vector&)>;

/**
 * @brief A neural network class.
 *
 * Construction of the neural network is set by passing `WeightVendor` instance to the constructor,
 * where wheight matrices and biases are initialized with this instance. The `DataVendor` instance provides
 * input data and target output for training; it can be changed later using `set_dv()` method.
 * 
 * @note The Network does NOT take ownership of the pointer. 
 *       The caller must ensure that `dv` outlives the Network instance.
 *
 * Is is mandatory to somehow initialize weight matrices, while biases are optional: by default
 * (if biases were not provided in `WeightVendor` instance), every bias vector that corresponds to the
 * specific layer in the neural network is set to zero vector.
 *
 * @note Sizes of the layers of the neural network are assumed to correspond with the weight matrices that
 * extracted from `WeightVendor` instance, it means that dimentions of the input and output vectors in the
 * `DataVendor` instance MUST be set properly.
 *
 * `layers[0]` is the input layer, `layer[n_layers - 1]` is the output layer,
 * the target vector is stored as `y`.
 *
 * Each i-th instance of the `Weight` stored in `weights` corresponds to the relationship between
 * (i-1)-th and i-th layers.
 *
 * Parameters such as the loss function partial derivative (`lp`), learning rate (`lr`),
 * and use bias flag (`wb`) can be adjusted as needed using the corresponding methods
 * (`set_lp()`, `set_lr()`, `set_wb()`). Default values are:
 * - `lp` = `mse_lp`
 * - `lr` = `0.1f`
 * - `wb` = `false`
 *
 * Activation functions (and their derivatives) can be set for each layer individually using
 * the `set_layer_activation()` method.
 */
class Network {
private:
    size_t n_layers;            // number of layers
    Layer* layers;              // layers
    Weight* weights;            // weights of layers (weight matrices and biases)
    Weight* grads;              // gradients
    Vector y;                   // target output vector
    Loss lp = mse_lp;           // partial derivative of the loss function with respect to the output layer
    float lr = 0.1f;            // learning rate
    bool wb = false;            // use bias flag
    DataVendor* dv = nullptr;   // pointer to data vendor

public:
    /**
     * @brief Constructs a new neural network.
     * @param wv `WeightVendor` used to construct neural network.
     * @param dv `DataVendor` supplying data for neural network.
     * 
     * @note The Network does NOT take ownership of the pointer.
     *       The caller must ensure that `dv` outlives the Network instance.
     */
    Network(const WeightVendor& wv, DataVendor* dv);
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    ~Network();

    /**
     * @brief Fills the input layer and target output vector with data from the vendor.
     * @param idx Index of the data sample to fetch.
     */
    void vend_data(size_t idx);

    /**
     * @brief Applies forward propagation.
     */
    void propagate();

    /**
     * @brief Applies backpropagation.
     */
    void backpropagate();

    /**
     * @brief Updates weight matrices and biases using computed gradients.
     */
    void apply_grads();

    /**
     * @brief Trains the network for a given number of epochs.
     * @param epochs Number of epochs to train.
     * 
     * During one epoch, the neural network is trained on the entire dataset once,
     * processing each data sample exactly once.
     */
    void epochs(size_t epochs);

    /**
     * @brief Computes the network output for a given input vector.
     * @param v Input vector for which to compute the result.
     * @return Output vector produced by the network.
     */
    Vector compute(const Vector& v);

    /**
     * @brief Sets the partial derivative of the loss function.
     * @param lp Partial derivative of the loss function with respect to the output layer.
     * @return Reference to this instance for method chaining.
     */
    Network& set_lp(const Loss& lp);

    /**
     * @brief Sets the learning rate.
     * @param lr The learning rate value.
     * @return Reference to this instance for method chaining.
     */
    Network& set_lr(float lr);

    /**
     * @brief Configures whether to use biases.
     * @param wb Whether to use biases in the neural network weights.
     * @return Reference to this instance for method chaining.
     */
    Network& set_wb(bool wb);

    /**
     * @brief Sets the data vendor.
     * @param dv Pointer to a `DataVendor` instance.
     * @return Reference to this instance for method chaining.
     * 
     * @note The Network does NOT take ownership of the pointer.
     *       The caller must ensure that `dv` outlives the Network instance.
     * 
     * The data vendor provides input data and target output for training.
     */
    Network& set_dv(DataVendor* dv);

    /**
     * @brief Sets the activation function and its derivative for a specific layer.
     * @param idx Index of the layer.
     * @param activation Activation function and its derivative.
     * @return Reference to this instance for method chaining.
     */
    Network& set_layer_activation(size_t idx, const Activation& activation);

    /**
     * @brief Layer getter.
     * @param idx Index of the layer to get.
     * @return Reference to the requested layer.
     */
    Layer& get_layer(size_t idx);

    /**
     * @brief Weight getter.
     * @param idx Index of the weight to get.
     * @return Reference to the requested weight.
     */
    Weight& get_weight(size_t idx);
};