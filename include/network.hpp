#pragma once

#include <cstddef>
#include <string>
#include <random>
#include "utils.hpp"

using Loss = std::function<Vector(const Vector&, const Vector&)>;

class Network {
private:
    size_t n_layers;
    Layer* layers;
    Weight* weights;
    Weight* grads;
    Weight* accumulated_grads;
    Loss lp;
    Vector out;

    size_t accumulation_steps;
    size_t current_accumulation;
    std::mt19937 rng;

public:
    Network(size_t, size_t, size_t, size_t, size_t accumulation_steps = 1);
    ~Network();
    void fill_from_path(std::string);
    void propagate(bool);
    void backpropagate(bool);
    void set_lp(const Loss&);
    Layer& get_layer(size_t idx);
    Weight& get_weight(size_t idx);

    void zero_accumulated_grads();
    void accumulate_current_grads();
    void apply_accumulated_grads(double lr);

    double train_sample(const Matrix& input, const Matrix& target,
        double lr, bool do_update = true);

    double train_epoch(const Matrix& data, const Matrix& labels,
        const Matrix& val_data, const Matrix& val_labels,
        size_t effective_batch_size, double lr);

    double validate(const Matrix& data, const Matrix& labels);

    void shuffle_indices(std::vector<size_t>& indices);
    void set_accumulation_steps(size_t steps);
    size_t get_accumulation_steps() const;
};