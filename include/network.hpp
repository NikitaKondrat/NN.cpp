#pragma once

#include <cstddef>
#include <string>
#include <random>
#include "utils.hpp"

using Loss = std::function<Vector(const Vector&, const Vector&)>;

/**
 * @brief A neural network class.
 * 
 * @warning Before training (i.e. using the `epochs()` method), a neural network 
 *       MUST be initialized with:
 *       1. Weight matrices
 *       2. A pointer to a `DataVendor` instance
 * 
 *       This is done using the `fill_weights()` and `set_dv()` methods, respectively.
 *       Using the `random_uniform_filler()` function from `utils.hpp` for weight 
 *       initialization is recommended.
 * 
 *       Furthermore, if bias will not be used in the training, it MUST be set to `false`
 *       with `set_wb()` method BEFORE filling weights of a network.
 * 
 * `layers[0]` is the input layer, `layer[n_layers - 1]` is the output layer, 
 * the target vector is stored as `y`.
 * 
 * Each i-th instance of the `Weight` stored in `weights` corresponds to the 
 * relationship between (i-1)-th and i-th layers.
 * 
 * Parameters such as the loss function partial derivative (`lp`), learning rate (`lr`), 
 * and use bias flag (`wb`) can be adjusted as needed using the corresponding methods 
 * (`set_lp()`, `set_lr()`, `set_wb()`). Default values are:
 * - `lp` = `mse_lp`
 * - `lr` = `0.1f`
 * - `wb` = `true`
 * 
 * Activation functions (and their derivatives) can be set for each layer using 
 * the `set_layer_activation()` method.
 */
class Network {
private:
    size_t n_layers;           // number of layers
    Layer* layers;             // layers
    Weight* weights;           // weights of layers (weight matrices and biases)
    Weight* grads;             // gradients
    Vector y;                  // target output vector
    Loss lp = mse_lp;          // partial derivative of the loss function with respect to the output layer
    float lr = 0.1f;           // learning rate
    bool wb = true;           // use bias flag
    DataVendor* dv = nullptr;  // pointer to data vendor

public:
    /**
     * @brief Constructs a new neural network.
     * @param n_layers Number of layers in the neural network, including input and output layers.
     * `size` below refers to the number of neurons in a specific layer.
     * @param l_size Number of neurons in each hidden layer.
     * @param in_size Number of neurons in the input layer.
     * @param out_size Number of neurons in the output layer.
     */
    Network(
        size_t n_layers, size_t l_size, 
        size_t in_size, size_t out_size
    );
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
     * The data vendor provides input data and target output for training.
     * @note The Network does NOT take ownership of the pointer.
     *       The caller must ensure that `dv` outlives the Network instance.
     */
    Network& set_dv(DataVendor* dv);

    /** 
     * @brief Initializes weight matrices using provided weight vendor.
     * @param wv Weight vendor.
     * @return Reference to this instance for method chaining.
     */
    Network& fill_weights(WeightVendor* wv);

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