#include "vendors.hpp"
#include <fstream>
#include <string>
#include <algorithm>
#include <stdexcept>

DataVendor::~DataVendor() {
    delete[] data;
    data = nullptr;
}

const Data& DataVendor::fetch(size_t idx) const {
    return data[idx];
}

size_t DataVendor::in_size() const noexcept {
    return in_size_;
}

size_t DataVendor::out_size() const noexcept {
    return out_size_;
}

FileDataVendor::FileDataVendor(const std::string& path) {
    std::ifstream ifs(path);
    ifs >> count_ >> in_size_ >> out_size_;
    data = new Data[count_];
    for (size_t i{}; i < count_; ++i) {
        Vector in(in_size_);
        Vector out(out_size_);
        float* in_data = in.data();
        float* out_data = out.data();
        for (size_t j{}; j < in_size_; ++j)
            ifs >> in_data[j];
        for (size_t j{}; j < out_size_; ++j)
            ifs >> out_data[j];
        data[i] = {std::move(in), std::move(out)};
    }
}

ObjectDataVendor::ObjectDataVendor(std::initializer_list<Data> l) {
    count_ = l.size();
    in_size_ = l.begin()->first.size();
    out_size_ = l.begin()->second.size();
    data = new Data[count_];
    std::copy(l.begin(), l.end(), data);
}

WeightVendor::~WeightVendor() {
    delete[] weights;
    weights = nullptr;
}

const Weight& WeightVendor::fetch(size_t idx) const {
    return weights[idx];
}

FileWeightVendor::FileWeightVendor(const std::string& path) {
    std::ifstream ifs(path);
    bool with_bias;
    ifs >> count_ >> with_bias;
    weights = new Weight[count_];

    for (size_t l{}; l < count_; ++l) {
        size_t r, c;
        ifs >> r >> c;
        Matrix weight(r, c);
        for (size_t i{}; i < r; ++i) {
            float* row_data = weight[i].data();
            for (size_t j{}; j < c; ++j)
                ifs >> row_data[j];
        }
        weights[l].w() = std::move(weight);

        if (with_bias) {
            Vector bias(r);
            float* bias_data = bias.data();
            for (size_t i{}; i < r; ++i)
                ifs >> bias_data[i];
            weights[l].b() = std::move(bias);
        } else {
            weights[l].b() = Vector(r);
        }
    }
}

ObjectWeightVendor::ObjectWeightVendor(std::initializer_list<Matrix> l) {
    count_ = l.size();
    weights = new Weight[count_];
    size_t i{};
    for (const auto& weight : l) {
        weights[i].w() = weight;
        weights[i].b() = Vector(weight.rows());
        ++i;
    }
}

ObjectWeightVendor::ObjectWeightVendor(std::initializer_list<std::pair<Matrix, Vector>> l) {
    count_ = l.size();
    weights = new Weight[count_];
    size_t i{};
    for (const auto& weight: l) {
        weights[i].w() = weight.first;
        weights[i].b() = weight.second;
        ++i;
    }
}

RandomWeightVendor::RandomWeightVendor(
    size_t n_layers, size_t in_size, size_t l_size,  size_t out_size, 
    bool with_bias, float a, float b
) : gen(std::random_device{}()), dist(a, b) {
    count_ = n_layers;
    FtoF func = [this](float) mutable -> float { return dist(gen); };
    weights = new Weight[n_layers - 1];

    weights[0].w() = Matrix(l_size, in_size).map(func);
    for (size_t i{1}; i < n_layers - 2; ++i) 
        weights[i].w() = Matrix(l_size, l_size).map(func);
    weights[n_layers - 2].w() = Matrix(out_size, l_size).map(func);

    for (size_t i{}; i < n_layers - 2; ++i) {
        if (with_bias)
            weights[i].b() = Vector(l_size).map(func);
        else
            weights[i].b() = Vector(l_size);
    }
    if (with_bias)
        weights[n_layers - 2].b() = Vector(out_size).map(func);
    else
        weights[n_layers - 2].b() = Vector(out_size);
}

ActivationVendor::ActivationVendor(size_t n_layers) {
    count_ = n_layers;
    activations = new Activation[n_layers];
}

ActivationVendor::~ActivationVendor() {
    delete[] activations;
    activations = nullptr;
}

Activation& ActivationVendor::fetch(size_t idx) const {
    return activations[idx];
}

ActivationVendor& ActivationVendor::set_hid(const Activation& activation) {
    for (size_t i{1}; i < count_ - 1; ++i)
        activations[i] = activation;
    return *this;
}

ActivationVendor& ActivationVendor::set_out(const Activation& activation) {
    activations[count_ - 1] = activation;
    return *this;
}

ActivationVendor& ActivationVendor::set_l(size_t idx, const Activation& activation) {
    if (idx >= count_)
        throw std::out_of_range("Activation index out of range");
    activations[idx] = activation;
    return *this;
}
