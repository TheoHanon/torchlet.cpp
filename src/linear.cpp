#include "detail/helpers.h"

#include <torchlet/module/linear.h>
#include <torchlet/ops/functional.h>
#include <torchlet/ops/init.h>

using torchlet::module::Linear, torchlet::core::Dtype, torchlet::core::Tensor,
    torchlet::core::Generator;

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias,
               const Dtype &dtype)
    : in_features(in_features), out_features(out_features), m_has_bias(bias) {

  if (dtype != Dtype::Float32 && dtype != Dtype::Float64) {
    throw std::invalid_argument(
        "Invalid input type. Only support float32 or float64.");
  }
  if (in_features <= 0 || out_features <= 0) {
    throw std::invalid_argument(
        "in_features and out_features must be positive.");
  }

  std::vector<std::size_t> shape_w{out_features, in_features};
  m_weights = Tensor(shape_w, dtype);

  DISPATCH_FLOAT(dtype, scalar_t, {
    torchlet::ops::init::uniform_(m_weights,
                                  -std::sqrt(scalar_t{1} / in_features),
                                  std::sqrt(scalar_t{1} / in_features));
  });

  if (bias) {
    std::vector<std::size_t> shape_b{out_features};
    m_bias = Tensor(shape_b, dtype);
    DISPATCH_FLOAT(dtype, scalar_t, {
      torchlet::ops::init::uniform_(m_bias,
                                    -std::sqrt(scalar_t{1} / in_features),
                                    std::sqrt(scalar_t{1} / in_features));
    });
  }

  return;
};

template <typename T> void Linear::normal_(T mean, T stdev, Generator &gen) {
  torchlet::ops::init::normal_(m_weights, mean, stdev, gen);
};

template <typename T> void Linear::uniform_(T start, T end, Generator &gen) {
  torchlet::ops::init::uniform_(m_weights, start, end, gen);
};

template void Linear::normal_(float, float, Generator &);
template void Linear::normal_(double, double, Generator &);

template void Linear::uniform_(float, float, Generator &);
template void Linear::uniform_(double, double, Generator &);

// naive implementation
Tensor Linear::forward(const Tensor &x) const {
  return torchlet::ops::linear(x, m_weights, m_bias);
}