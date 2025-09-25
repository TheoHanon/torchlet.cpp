#ifndef TENSOR_HELPERS_H
#define TENSOR_HELPERS_H

#include <torchcpp/dtype.h>
#include <vector>

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

inline std::vector<std::size_t> get_strides(std::vector<std::size_t> shape) noexcept {
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;

    for (std::size_t i = shape.size(); i > 0 ; --i){
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
};

inline constexpr std::size_t numel(const std::vector<std::size_t>& shape) noexcept {
    std::size_t numel = 1;
    for (const auto& s: shape) numel *= s;
    return numel;
};

inline constexpr std::size_t nbytes(const std::vector<std::size_t>& shape, Dtype dtype) {
    return dtype_size(dtype) * numel(shape);
};


#endif //TENSOR_HELPERS