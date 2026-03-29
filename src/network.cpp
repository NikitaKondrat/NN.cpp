#include "network.hpp"
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iostream>

Network::Network(
    size_t n_layers, size_t l_size, 
    size_t in_size, size_t out_size,
    size_t accumulation_steps
) {
    this->n_layers = n_layers;
    this->accumulation_steps = accumulation_steps;
    this->current_accumulation = 0;
    this->rng.seed(std::random_device{}());
    
    grads = new Weight[n_layers - 1];
    accumulated_grads = new Weight[n_layers - 1];

    layers = new Layer[n_layers];
    for (size_t i{1}; i < n_layers - 1; ++i)
        layers[i].set_z(Vector(l_size));
    layers[0].set_z(Vector(in_size));
    layers[n_layers - 1].set_z(Vector(out_size));

    weights = new Weight[n_layers - 1];
    for (size_t i{1}; i < n_layers - 2; ++i)
        weights[i].set_w(Matrix(l_size, l_size));
    weights[0].set_w(Matrix(l_size, in_size));
    weights[n_layers - 2].set_w(Matrix(out_size, l_size));

    for (size_t i{ 0 }; i < n_layers - 1; ++i) {
        if (i < n_layers - 2) {
            size_t rows = (i == n_layers - 2) ? out_size : l_size;
            size_t cols = (i == 0) ? in_size : l_size;
            accumulated_grads[i].set_w(Matrix(rows, cols));
            accumulated_grads[i].set_b(Vector(rows));
        }
        else {
            accumulated_grads[i].set_w(Matrix(out_size, l_size));
            accumulated_grads[i].set_b(Vector(out_size));
        }
    }
}

Network::~Network() {
    delete[] layers;
    delete[] weights;
    delete[] grads;
    delete[] accumulated_grads;
}

void Network::zero_accumulated_grads() {
    for (size_t i = 0; i < n_layers - 1; ++i) {
        const Matrix& old_w = accumulated_grads[i].w();
        Matrix zero_w(old_w.rows(), old_w.cols());
        accumulated_grads[i].set_w(zero_w);

        const Vector& old_b = accumulated_grads[i].b();
        Vector zero_b(old_b.size());
        accumulated_grads[i].set_b(zero_b);
    }
    current_accumulation = 0;
}

void Network::accumulate_current_grads() {
    for (size_t i = 0; i < n_layers - 1; ++i) {
        const Matrix& acc_w = accumulated_grads[i].w();
        const Matrix& cur_w = grads[i].w();

        Matrix new_w(acc_w.rows(), acc_w.cols());
        for (size_t r = 0; r < acc_w.rows(); ++r) {
            const float* acc_row = acc_w.data()[r].data();
            const float* cur_row = cur_w.data()[r].data();
            float* new_row = new_w.data()[r].data();
            for (size_t c = 0; c < acc_w.cols(); ++c)
                new_row[c] = acc_row[c] + cur_row[c];
        }

        const Vector& acc_b = accumulated_grads[i].b();
        const Vector& cur_b = grads[i].b();

        Vector new_b(acc_b.size());
        for (size_t j = 0; j < acc_b.size(); ++j)
            new_b[j] = acc_b[j] + cur_b[j];

        accumulated_grads[i].set_w(new_w);
        accumulated_grads[i].set_b(new_b);
    }
    current_accumulation++;
}

void Network::apply_accumulated_grads(double lr) {
    if (current_accumulation == 0)
        return;

    double scale = lr / static_cast<double>(current_accumulation);

    for (size_t i = 0; i < n_layers - 1; ++i) {
        const Matrix& w = weights[i].w();
        const Matrix& grad_w = accumulated_grads[i].w();

        Matrix new_w(w.rows(), w.cols());
        for (size_t r = 0; r < w.rows(); ++r) {
            const float* w_row = w.data()[r].data();
            const float* grad_row = grad_w.data()[r].data();
            float* new_row = new_w.data()[r].data();
            for (size_t c = 0; c < w.cols(); ++c)
                new_row[c] = static_cast<float>(w_row[c] - scale * grad_row[c]);
        }

        const Vector& b = weights[i].b();
        const Vector& grad_b = accumulated_grads[i].b();

        Vector new_b(b.size());
        for (size_t j = 0; j < b.size(); ++j)
            new_b[j] = static_cast<float>(b[j] - scale * grad_b[j]);

        weights[i].set_w(new_w);
        weights[i].set_b(new_b);
    }

    current_accumulation = 0;
    zero_accumulated_grads();
}

double Network::train_sample(const Matrix& input, const Matrix& target,
    double lr, bool do_update) {
    if (input.rows() != 1)
        throw std::runtime_error("Input must be a single sample (1 row)");

    const Row& input_row = input[0];
    Vector input_vec(input_row.size());
    const float* input_data = input_row.data();
    for (size_t j = 0; j < input_row.size(); ++j)
        input_vec[j] = input_data[j];

    layers[0].set_z(input_vec);
    propagate(true);

    Vector output = layers[n_layers - 1].gz();
    Vector target_vec(target.cols());
    const Row& target_row = target[0];
    const float* target_data = target_row.data();
    for (size_t j = 0; j < target.cols(); ++j)
        target_vec[j] = target_data[j];

    Vector loss_vec = lp(output, target_vec);
    double loss = 0.0;
    const float* loss_data = loss_vec.data();
    for (size_t j = 0; j < loss_vec.size(); ++j)
        loss += loss_data[j];
    loss /= loss_vec.size();

    backpropagate(true);
    accumulate_current_grads();

    if (do_update && current_accumulation >= accumulation_steps) {
        apply_accumulated_grads(lr);
    }

    return loss;
}

void Network::shuffle_indices(std::vector<size_t>& indices) {
    std::shuffle(indices.begin(), indices.end(), rng);
}

double Network::train_epoch(const Matrix& data, const Matrix& labels,
    const Matrix& val_data, const Matrix& val_labels,
    size_t effective_batch_size, double lr) {
    size_t n_samples = data.rows();
    if (n_samples != labels.rows())
        throw std::runtime_error("Data and labels must have same number of samples");

    size_t batch_size = effective_batch_size / accumulation_steps;
    if (batch_size == 0) batch_size = 1;

    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    shuffle_indices(indices);

    double total_loss = 0.0;
    size_t samples_processed = 0;

    zero_accumulated_grads();

    for (size_t batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
        size_t current_batch_size = std::min(batch_size, n_samples - batch_start);
        bool is_last_batch = (batch_start + current_batch_size >= n_samples);

        for (size_t i = 0; i < current_batch_size; ++i) {
            size_t idx = indices[batch_start + i];

            Matrix sample_input(1, data.cols());
            const Row& data_row = data[idx];
            const float* data_row_ptr = data_row.data();
            float* sample_input_row = sample_input[0].data();
            for (size_t j = 0; j < data.cols(); ++j)
                sample_input_row[j] = data_row_ptr[j];

            Matrix sample_target(1, labels.cols());
            const Row& label_row = labels[idx];
            const float* label_row_ptr = label_row.data();
            float* sample_target_row = sample_target[0].data();
            for (size_t j = 0; j < labels.cols(); ++j)
                sample_target_row[j] = label_row_ptr[j];

            double loss = train_sample(sample_input, sample_target, lr, !is_last_batch);
            total_loss += loss;
            samples_processed++;
        }

        if (current_accumulation >= accumulation_steps || is_last_batch) {
            apply_accumulated_grads(lr);
        }
    }

    double val_loss = validate(val_data, val_labels);

    std::cout << "Epoch complete - Train Loss: " << (total_loss / samples_processed)
        << ", Val Loss: " << val_loss
        << ", Samples: " << samples_processed << std::endl;

    return total_loss / samples_processed;
}

double Network::validate(const Matrix& data, const Matrix& labels) {
    size_t n_samples = data.rows();
    double total_loss = 0.0;

    for (size_t i = 0; i < n_samples; ++i) {
        const Row& data_row = data[i];
        const float* data_row_ptr = data_row.data();
        Vector input_vec(data_row.size());
        for (size_t j = 0; j < data_row.size(); ++j)
            input_vec[j] = data_row_ptr[j];

        layers[0].set_z(input_vec);
        propagate(true);

        Vector output = layers[n_layers - 1].gz();
        const Row& label_row = labels[i];
        const float* label_row_ptr = label_row.data();
        Vector target_vec(label_row.size());
        for (size_t j = 0; j < label_row.size(); ++j)
            target_vec[j] = label_row_ptr[j];

        Vector loss_vec = lp(output, target_vec);
        double sample_loss = 0.0;
        const float* loss_data = loss_vec.data();
        for (size_t j = 0; j < loss_vec.size(); ++j)
            sample_loss += loss_data[j];
        sample_loss /= loss_vec.size();

        total_loss += sample_loss;
    }

    return total_loss / n_samples;
}

void Network::set_accumulation_steps(size_t steps) {
    if (steps == 0)
        throw std::invalid_argument("Accumulation steps must be > 0");
    accumulation_steps = steps;
}

size_t Network::get_accumulation_steps() const {
    return accumulation_steps;
}

void Network::fill_from_path(std::string path) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("couldn't open file");

    size_t in_size = layers[0].z().size();
    Vector in_data(in_size);
    for (size_t i = 0; i < in_size; ++i)
        in >> in_data[i];
    layers[0].set_z(std::move(in_data));

    size_t out_size = layers[n_layers - 1].z().size();
    for (size_t i = 0; i < out_size; ++i)
        in >> out[i];

    in.close();
}

void Network::propagate(bool with_bias = false) {
    for (size_t i{}; i < n_layers - 1; ++i) {
        const Weight& weight = weights[i];
        Vector z = weight.w() * layers[i].gz();
        const Vector& b = weight.b();
        if (with_bias)
            z += weight.b();
        layers[i + 1].set_z(z);
    }
}

void Network::backpropagate(bool with_out_activ = true) {
    size_t k = n_layers - 1;
    Layer& L = layers[k];
    Layer& L_p = layers[k - 1];

    Vector dl_dL, dL_dz, dl_dz;
    if (with_out_activ) {
        dl_dL = lp(L.gz(), out);
        dL_dz = L.z().map(L.activ().activ_deriv);
        dl_dz = hadamar(dl_dL, dL_dz);
    } else
        dl_dz = lp(L.gz(), out);

    grads[k - 1].set_w(outer_product(dl_dz, L_p.gz()));
    grads[k - 1].set_b(dl_dz);

    for (k = n_layers - 2; k > 0; --k) {
        L = layers[k];
        L_p = layers[k - 1];

        dl_dL = dl_dz * weights[k].w();
        dL_dz = L.z().map(L.activ().activ_deriv);
        dl_dz = hadamar(dl_dL, dL_dz);

        grads[k - 1].set_w(outer_product(dl_dz, L_p.gz()));
        grads[k - 1].set_b(dl_dz);
    }
}

void Network::set_lp(const Loss& lp) {
    this->lp = lp;
}

Layer& Network::get_layer(size_t idx) {
    if (idx >= n_layers)
        throw std::out_of_range("layer index out of range");
    return layers[idx];
}

Weight& Network::get_weight(size_t idx) {
    if (idx >= n_layers - 1)
        throw std::out_of_range("weight index out of range");
    return weights[idx];
}
