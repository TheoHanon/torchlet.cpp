#pragma once
#include <vector>
#include <stdexcept>

namespace torchlet::detail {

void validate_shape(std::vector<size_t> old_shape, std::vector<size_t> new_shape){

    size_t new_prod = 1; size_t old_prod = 1;

    for (size_t k = 0 ; k < new_shape.size(); k++) new_prod *= new_shape[k];
    for (size_t k = 0 ; k < old_shape.size(); k++) old_prod *= old_shape[k];

    if (old_prod != new_prod) throw std::invalid_argument("Shapes doesn't match.");
};

void validate_strides(const std::vector<size_t>& strides) {

    for (size_t i = 1; i < strides.size(); i++) {
        if (strides[i - 1] < strides[i]) throw std::runtime_error("Memory layout is not row major.");
    }
};

}