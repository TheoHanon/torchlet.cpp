#pragma once
#include <iostream>
#include <torchlet/core/dtype.h>
#include <vector>

namespace torchlet::detail {

inline constexpr std::size_t dtype_size(torchlet::core::Dtype dtype) noexcept {
  switch (dtype) {
  case torchlet::core::Dtype::Float32:
    return 4;
  case torchlet::core::Dtype::Float64:
    return 8;
  case torchlet::core::Dtype::Int32:
    return 4;
  case torchlet::core::Dtype::Int64:
    return 8;
  case torchlet::core::Dtype::UInt8:
    return 1;
  case torchlet::core::Dtype::UInt32:
    return 4;
  case torchlet::core::Dtype::UInt64:
    return 8;
  }
  return 0;
};

inline size_t get_offset(const std::initializer_list<size_t> &index,
                         const std::vector<size_t> &shape,
                         const std::vector<size_t> &strides,
                         const size_t &curr_offset) {

  size_t offset = curr_offset;
  size_t k = 0;

  for (auto &id : index) {
    if (id >= shape[k])
      throw std::invalid_argument("Index out of range.");

    offset += strides[k++] * id;
  }

  return offset;
}

inline std::vector<std::size_t>
get_strides(const std::vector<std::size_t> &shape) noexcept {
  std::vector<std::size_t> strides(shape.size());
  std::size_t stride = 1;

  size_t idx = shape.size();

  for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
    strides[--idx] = stride;
    stride *= *it;
  }

  return strides;
};

inline std::size_t numel(const std::vector<std::size_t> &shape) noexcept {
  std::size_t numel = 1;
  for (const auto &s : shape)
    numel *= s;
  return numel;
};

inline std::size_t nbytes(const std::vector<std::size_t> &shape,
                          torchlet::core::Dtype dtype) {
  return dtype_size(dtype) * numel(shape);
};

} // namespace torchlet::detail
