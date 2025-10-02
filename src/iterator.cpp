#include "detail/helpers.h"
#include <stack>
#include <torchlet/iterator/iterator.h>

using torchlet::core::Tensor, torchlet::iterator::ContiguousIterator,
    torchlet::core::Dtype;

ContiguousIterator::ContiguousIterator(
    Tensor *out, std::initializer_list<const Tensor *const> inputs) {

  std::vector<std::size_t> out_shape = out->shape();

  output_dim = out_shape.back();
  itemsize = torchlet::detail::dtype_size(out->dtype());
  batch_size = 1;

  for (std::size_t k = 0; k < out_shape.size() - 1; k++)
    batch_size *= out_shape[k];

  output_ptr = out->data_ptr<std::uint8_t>();

  if (inputs.size() != 0) {
    bool set = false;
    input_ptrs.reserve(inputs.size());
    for (auto &in : inputs) {
      if (!set) {
        input_dim = in->shape().back();
        set = true;
      }
      input_ptrs.push_back(in->data_ptr<std::uint8_t>());
    }
  }
};
