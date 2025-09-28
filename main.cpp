#include <torchlet/torchlet.h>

#include <bitset>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace torchlet::index;

template <typename T> void rprint(Tensor &tensor, size_t dim, size_t &index) {
  std::vector<size_t> shape = tensor.shape();
  T *data_ptr = tensor.data_ptr<T>() + tensor.elem_offset();

  if (dim == shape.size() - 1) {
    std::cout << "[ ";
    vector<size_t> strides = tensor.strides();

    for (size_t idx = 0; idx < shape[dim]; idx++) {
      size_t tmp = index;
      size_t offset = 0;
      for (size_t dim = shape.size(); dim-- > 0;) {
        size_t coord = tmp % shape[dim];
        tmp /= shape[dim];
        offset += coord * strides[dim];
      }
      std::cout << *(data_ptr + offset);
      if (idx != shape[dim] - 1)
        std::cout << ", ";
      index++;
    }
    std::cout << " ]";
  } else {
    std::cout << "[";
    for (size_t k = 0; k < shape[dim]; k++) {
      rprint<T>(tensor, dim + 1, index);
      if (k != shape[dim] - 1)
        std::cout << ", ";
    }
    std::cout << "]";
  }
};

template <typename T> void print(Tensor &tensor) {
  size_t index = 0;
  rprint<T>(tensor, 0, index);
  std::cout << std::endl;
};

template <typename T> void print_raw(Tensor &tensor) {
  size_t len = 1;
  std::vector<size_t> shape = tensor.shape();
  T *data_ptr = tensor.data_ptr<T>() + tensor.elem_offset();

  for (const auto &sh : shape)
    len *= sh;
  std::cout << "[";
  for (size_t k = 0; k < len; k++) {
    std::cout << *(data_ptr + k) << " ";
  }

  std::cout << "]" << std::endl;
}

int main() {
  size_t in_features = 10;
  size_t out_features = 5;

  Linear linear(in_features, out_features, false, Dtype::Float32);
  linear.uniform_(1.0f, 1.0f);

  Tensor x = Tensor::ones({5, 10}, Dtype::Float32);

  // torchlet::init::uniform_(x, 0.0f, 1.0f);

  Tensor out = linear.forward(x);

  print<float>(out);

  // vector<size_t> shape{2, 3, 5};
  // Dtype dtype = Dtype::Float32;

  // Tensor t1 = Tensor(shape, dtype);
  // Tensor t2 = t1.index({Slice(0, 2), Slice(0, 3), Slice(0, 5)});

  // print<float>(t1);
  // print<float>(t2);

  // torchlet::init::normal_(t2, 0.f, 1.f);

  // print<float>(t1);
  // print<float>(t2);

  // std::cout << "\n\n";

  return 0;
}