#pragma once

#include "utils.hpp"
#include <cstddef>
#include <initializer_list>

/**
 * @brief Base abstract class for all resource vendors.
 * 
 * Vendors provide external resources (data, weights, etc.) to a neural network,
 * separating resource acquisition from core training logic.
 */
template<typename T>
class Vendor {
protected:
    size_t count_ = 0; // number of items provided by vendor
public:
    Vendor() = default;
    virtual ~Vendor() = default;
    Vendor(const Vendor&) = delete;
    Vendor& operator=(const Vendor&) = delete;

    /**
     * @brief Fetches a single item.
     * @warning If `idx` >= `count_` it is an UB.
     * @param idx Zero-based index of the item.
     * @return Constant reference to the requested item.
     */
    virtual const T& fetch(size_t idx) const = 0;

    /**
     * @brief Number of stored items getter.
     * @return Number of stored items.
     */
    size_t count() const noexcept {
        return count_;
    };
};

/**
 * @brief Base class for neural network data providers.
 * 
 * @note This class itself is not used to construct instances of data providers,
 *       use derived classes instead. All restrictions, however, MUST be observed in every derived class.
 * 
 * @warning When used with a neural network:
 *          - `in_size_` MUST be equal to the input layer size (`layers[0].z().size()`).
 *          - `out_size_` MUST be equal to the target vector size (`y.size()`).
 */
class DataVendor : public Vendor<Data> {
protected:
    size_t in_size_ = 0;   // input vector size
    size_t out_size_ = 0;  // target vector size
    Data* data = nullptr;  // stored data
                           // `count_` for this class means dataset size
    DataVendor() = default;
    virtual ~DataVendor();
public:
    DataVendor(const DataVendor&) = delete;
    DataVendor& operator=(const DataVendor&) = delete;  

    /**
     * @brief Fetches a single sample from the dataset.
     * @warning If `idx` >= `count_` it is an UB.
     * @param idx Zero-based index of the data sample to fetch from `data`.
     * @return Constant reference of a `std::pair<Data>` instance, where
     *         - first item is input vector data.
     *         - second item is target vector data.
     */
    virtual const Data& fetch(size_t idx) const override;

    /**
     * @brief Input vector size getter.
     * @return Input vector size.
     */
    
    size_t in_size() const noexcept;

    /**
     * @brief Target vector size getter.
     * @return Target vector size. 
     */
    size_t out_size() const noexcept;
};

/**
 * @brief Class of a data provider that loads data from file.
 * 
 * Values are read using standard stream extraction (`>>`), so they may be 
 * separated by any whitespace (spaces, tabs, or newlines).
 * 
 * @par Expected File Structure:
 * 1. **Header**:
 *    - `count` (size_t)    : Total number of training samples.
 *    - `in_size` (size_t)  : Dimension of each input vector.
 *    - `out_size` (size_t) : Dimension of each target vector.
 * 2. **Sample Blocks** (repeated exactly `count` times):
 *    - `in_size` `out_size` values : Elements of the input and output vectors.
 * 
 * @param path Absolute or relative path to the dataset file.
 */
class FileDataVendor : public DataVendor {
public:
    FileDataVendor(const std::string& path);
};

/**
 * @brief Class of a data provider that stores data copied from an initializer list.
 */
class ObjectDataVendor : public DataVendor {
public:
    ObjectDataVendor(std::initializer_list<Data> l);
};

/**
 * @brief Base class for neural network weights providers.
 * 
 * @note This class itself is not used to construct instances of weights providers,
 *       use derived classes instead. All restrictions, however, MUST be observed in every derived class.
 * 
 * @warning When used with a neural network:
 *          - `weights[0].w().cols()` MUST be equal to `layers[0].z().size()`.
 *          - `weights[count_ - 1].w().rows()` MUST be equal to `y.size()`.
 *          - `weights[i].b().size()` MUST be equal to the corresponding bias in a neural network.
 */
class WeightVendor : public Vendor<Weight> { 
protected:
    Weight* weights = nullptr;  // stored weight matrices and biases
    WeightVendor() = default;  
    virtual ~WeightVendor();
public:
    WeightVendor(const WeightVendor&) = delete;
    WeightVendor& operator=(const WeightVendor&) = delete;
    
    /**
     * @brief Fetches a single weight from the stored weights.
     * @warning If `idx` >= `count_` it is an UB.
     * @param idx Zero-based index of the weight to fetch from `weights`.
     * @return Constant reference of a `Weight` instance.
     */
    virtual const Weight& fetch(size_t idx) const override;
};

/**
 * @brief Class of a weight provider that loads data from file.
 * 
 * Values are read using standard stream extraction (`>>`), so they may be 
 * separated by any whitespace (spaces, tabs, or newlines).
 * 
 * @par Expected File Structure:
 * 1. **Header**:
 *    - `count` (size_t)    : Number of weight layers to load.
 *    - `with_bias` (bool)  : Non-zero if bias vectors are present in the file.
 * 2. **Layer Blocks** (repeated exactly `count` times):
 *    - `rows` `cols`       : Dimensions of the weight matrix.
 *    - `rows × cols` values: Weight matrix elements in **row-major** order.
 *    - `rows` values       : Bias vector elements (read only if `with_bias != 0`).
 * 
 * @param path Absolute or relative path to the dataset file.
 */
class FileWeightVendor : public WeightVendor {
public:
    FileWeightVendor(const std::string& path);
};

/**
 * @brief Class of a weight provider that stores data copied from an initializer list.
 */
class ObjectWeightVendor : public WeightVendor {
public:
    ObjectWeightVendor(std::initializer_list<Matrix> l);
    ObjectWeightVendor(std::initializer_list<std::pair<Matrix, Vector>> l);
};

/**
 * @brief Provider of a filler function for weights initialization in a neural network.
 * @param a Lower bound of the distribution.
 * @param b Upper bound of the distribution.
 * @return Filler function.
 */

/**
 * @brief Class of a weight provider that gives random weights.
 * @note Uses a Mersenne Twister generator to make weights.
 */
class RandomWeightVendor : public WeightVendor {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
public:
    /**
     * @brief Constructor.
     * @param n_layers Number of layers in a neural network.
     * @param in_size Size of the input layer in a neural network.
     * @param l_size Size of the hidden layer in a neural network.
     * @param out_size Size of the output layer in a neural network.
     * @param with_bias Use bias flag, default = `false`.
     * @param a Lower bound of the distribution, default = `-1.0f`.
     * @param b Upper bound of the distribution, default = `1.0f`.
     */
    RandomWeightVendor(
        size_t n_layers, size_t in_size, size_t l_size,  size_t out_size, bool with_bias = false,
        float a = -1.0f, float b = 1.0f
    );
};

/**
 * @brief Class of a vendor providing activation functions for each layer of a neural network.
 * 
 * - Index `0` corresponds to the input layer (usually identity).
 * - Indices `1..n_layers-2` correspond to hidden layers.
 * - Index `n_layers-1` corresponds to the output layer.
 */
class ActivationVendor : public Vendor<Activation> {
private:
    Activation* activations;    // stored activation functions and their derivatives
                                // `count_` for this class means number of layers in a neural network
public:
    /**
     * @brief Constructor.
     * @param n_layers Number of layers in a neural network.
     * 
     * @warning `n_layers` CANNOT be less, than 3.
     */
    ActivationVendor(size_t n_layers);
    ActivationVendor(const ActivationVendor&) = delete;
    ActivationVendor& operator=(const ActivationVendor&) = delete;
    ~ActivationVendor();

    /**
     * @brief Fetches an `Activation` at a specific index.
     * @warning If `idx` >= `count_` it is an UB.
     * @param idx Zero-based index.
     * @return Constant reference of an `Activation`.
     */
    Activation& fetch(size_t idx) const override;

    /**
     * @brief Sets the same activation function for all hidden layers (indices `1..n_layers-2`).
     * @param activation The `Activation` to apply to each hidden layer.
     * @return Reference to this instance for method chaining.
     */
    ActivationVendor& set_hid(const Activation& activation);

    /**
     * @brief Sets the activation function for the output layer (index `n_layers-1`).
     * @param activation The `Activation` to apply to the output layer.
     * @return Reference to this instance for method chaining.
     */
    ActivationVendor& set_out(const Activation& activation);

    /**
     * @brief Sets the activation function for a specific layer by zero-based index.
     * @param idx Zero-based layer index (0 ≤ idx < count_).
     * @param activation The `Activation` to assign to layer `idx`.
     * @return Reference to this instance for method chaining.
     */
    ActivationVendor& set_l(size_t idx, const Activation& activation);
};