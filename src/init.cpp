#include <torchlet/ops/init.h>

// TODO : Remove hard coded loops -> implement iterator.

using torchlet::core::Tensor, torchlet::core::Generator;

template <typename T>
void torchlet::ops::init::normal_(Tensor &tensor, T mean, T stdev,
                                  Generator &gen) {

  if (CPPTypeToDType<T>::dtype != tensor.dtype()) {
    throw std::runtime_error("Type T does not match the type of dtype.");
  }

  std::normal_distribution<T> dist{mean, stdev};

  std::size_t elem_offset = tensor.elem_offset();
  T *data_ptr = tensor.data_ptr<T>();

  if (tensor.is_contiguous()) {
    for (std::size_t idx = 0; idx < tensor.numel(); idx++)
      data_ptr[idx + elem_offset] = dist(gen.engine());
  } else {

    std::vector<std::size_t> shape = tensor.shape();
    std::vector<std::size_t> strides = tensor.strides();

    for (std::size_t idx = 0; idx < tensor.numel(); idx++) {
      std::size_t offset = elem_offset;
      std::size_t tmp = idx;
      for (std::size_t dim = shape.size(); dim-- > 0;) {
        std::size_t coord = tmp % shape[dim];
        tmp /= shape[dim];
        offset += coord * strides[dim];
      }
      data_ptr[offset] = dist(gen.engine());
    }
  }

  return;
};

template <typename T>
void torchlet::ops::init::uniform_(Tensor &tensor, T start, T end,
                                   Generator &gen) {

  if (CPPTypeToDType<T>::dtype != tensor.dtype()) {
    throw std::runtime_error("Type T does not match the type of dtype.");
  }

  std::uniform_real_distribution<T> dist{start, end};

  std::size_t elem_offset = tensor.elem_offset();
  T *data_ptr = tensor.data_ptr<T>();

  if (tensor.is_contiguous()) {
    for (std::size_t idx = 0; idx < tensor.numel(); idx++)
      data_ptr[idx + elem_offset] = dist(gen.engine());
  } else {

    std::vector<std::size_t> shape = tensor.shape();
    std::vector<std::size_t> strides = tensor.strides();

    for (std::size_t idx = 0; idx < tensor.numel(); idx++) {
      std::size_t offset = elem_offset;
      std::size_t tmp = idx;
      for (std::size_t dim = shape.size(); dim-- > 0;) {
        std::size_t coord = tmp % shape[dim];
        tmp /= shape[dim];
        offset += coord * strides[dim];
      }
      data_ptr[offset] = dist(gen.engine());
    }
  }
};

template void torchlet::ops::init::normal_(Tensor &, float, float, Generator &);
template void torchlet::ops::init::normal_(Tensor &, double, double,
                                           Generator &);

template void torchlet::ops::init::uniform_(Tensor &, float, float,
                                            Generator &);
template void torchlet::ops::init::uniform_(Tensor &, double, double,
                                            Generator &);
