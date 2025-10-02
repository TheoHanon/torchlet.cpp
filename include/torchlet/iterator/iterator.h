#pragma once
#include <cstddef>
#include <vector>

#include <torchlet/core/dtype.h>
#include <torchlet/core/tensor.h>

namespace torchlet::iterator {

struct ContiguousIterator {
  ContiguousIterator() = default;
  ContiguousIterator(
      torchlet::core::Tensor *out,
      std::initializer_list<const torchlet::core::Tensor *const> inputs);

  template <typename Lambda> void for_each(Lambda &&lambda);

  std::size_t itemsize{0};
  std::size_t input_dim{0};
  std::size_t output_dim{0};
  std::size_t batch_size{1};

  std::uint8_t *output_ptr = nullptr;
  std::vector<const std::uint8_t *> input_ptrs;

  template <typename Lambda> void for_each_no_inputs(Lambda &&lambda);
  template <typename Lambda> void for_each_with_inputs(Lambda &&lambda);
};

template <typename Lambda>
void ContiguousIterator::for_each_no_inputs(Lambda &&lambda) {
  std::size_t out_step = output_dim * itemsize;
  std::uint8_t *out_ptr = output_ptr;
  for (std::size_t b = 0; b < batch_size; ++b) {
    lambda(out_ptr);
    out_ptr += out_step;
  }
};

template <typename Lambda>
void ContiguousIterator::for_each_with_inputs(Lambda &&lambda) {
  std::size_t in_step = input_dim * itemsize;
  std::size_t out_step = output_dim * itemsize;

  std::uint8_t *out_ptr = output_ptr;
  std::vector<const std::uint8_t *> in_ptrs = input_ptrs;
  std::size_t in_size = in_ptrs.size();

  for (std::size_t b = 0; b < batch_size; ++b) {
    lambda(out_ptr, in_ptrs.data(), in_size);
    out_ptr += out_step;
    for (auto &ptr : in_ptrs)
      ptr += in_step;
  }
};

} // namespace torchlet::iterator
