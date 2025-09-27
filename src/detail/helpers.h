#pragma once
#include <torchlet/dtype.h>
#include <vector>
#include <iostream>

namespace torchlet::detail {


inline constexpr std::size_t dtype_size(Dtype dtype) noexcept {
    switch (dtype) {
        case Dtype::Float32: return 4;
        case Dtype::Float64: return 8;
        case Dtype::Int32:   return 4;
        case Dtype::Int64:   return 8;
        case Dtype::UInt8:   return 1;
    }
    return 0;
};

inline size_t get_offset(const std::initializer_list<size_t>& index, const std::vector<size_t>& shape, const std::vector<size_t>& strides, const size_t& curr_offset) {

    size_t offset = curr_offset;
    size_t k = 0;

    for (auto &id : index)
    {
        if (id >= shape[k])
            throw std::runtime_error("Index out of range.");

        offset += strides[k++] * id;
    }

    return offset;

}

inline std::vector<std::size_t> get_strides(const std::vector<std::size_t>& shape) noexcept {
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;

    size_t idx = shape.size();

    for (auto it = shape.rbegin(); it != shape.rend(); ++it){
        strides[--idx] = stride;
        stride *= *it;
    }

    return strides;
};

inline std::size_t numel(const std::vector<std::size_t>& shape) noexcept {
    std::size_t numel = 1;
    for (const auto& s: shape) numel *= s;
    return numel;
};

inline std::size_t nbytes(const std::vector<std::size_t>& shape, Dtype dtype) {
    return dtype_size(dtype) * numel(shape);
};

}
