#include <torchlet/torchlet.h>

#include <bitset>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace torchlet::core::index;
using torchlet::core::Tensor, torchlet::core::Dtype, torchlet::module::Linear,
    torchlet::core::Generator;

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
    std::cout << "]\n";
  }
};

template <typename T> void print(Tensor &tensor) {
  size_t index = 0;
  rprint<T>(tensor, 0, index);
  std::cout << "\n" << std::endl;
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

struct Module {

  size_t input_dim;
  size_t output_dim;
  size_t n_layer;
  size_t hidden_features;
  Dtype dtype;
  std::vector<Linear> modules;

  Module() = delete;

  Module(size_t input_dim, size_t output_dim, size_t n_layer,
         size_t hidden_features, Dtype dtype)
      : input_dim(input_dim), output_dim(output_dim), n_layer(n_layer),
        hidden_features(hidden_features), dtype(dtype) {

    modules.emplace_back(input_dim, hidden_features, false, dtype);
    for (auto k = 0; k < n_layer - 2; ++k)
      modules.emplace_back(hidden_features, hidden_features, false, dtype);

    modules.emplace_back(hidden_features, output_dim, false, dtype);
  };

  Tensor forward(Tensor &x) {
    Tensor out = x;
    for (auto k = 0; k < n_layer - 1; ++k) {
      auto mod = modules[k];
      out = mod.forward(out);
      out = torchlet::ops::gelu(out);
    }
    out = modules.back().forward(out);
    out = torchlet::ops::softmax(out);
    return out;
  }
};

int main() {

  Generator::global().manual_seed(42);
  Module nn(5, 10, 3, 10, Dtype::Float32);
  Tensor x({5}, Dtype::Float32);
  torchlet::ops::init::normal_(x, 0.0f, 1.0f);

  Tensor out = nn.forward(x);
  print<float>(out);

  return 0;
}