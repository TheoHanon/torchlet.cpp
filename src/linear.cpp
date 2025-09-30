#include "detail/helpers.h"
#include <torchlet/init.h>
#include <torchlet/linear.h>

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

  std::vector<std::size_t> shape_w{in_features, out_features};
  m_weights = Tensor(shape_w, dtype);

  DISPATCH_FLOAT(dtype, scalar_t, {
    torchlet::init::uniform_(m_weights, -sqrt(scalar_t{1} / in_features),
                             sqrt(scalar_t{1} / in_features));
  });

  if (bias) {
    std::vector<std::size_t> shape_b{out_features};
    m_bias = Tensor(shape_b, dtype);
    DISPATCH_FLOAT(dtype, scalar_t, {
      torchlet::init::uniform_(m_bias, -sqrt(scalar_t{1} / in_features),
                               sqrt(scalar_t{1} / in_features));
    });
  }

  return;
};

template <typename T> void Linear::normal_(T mean, T stdev, Generator &gen) {
  torchlet::init::normal_(m_weights, mean, stdev, gen);
};

template <typename T> void Linear::uniform_(T start, T end, Generator &gen) {
  torchlet::init::uniform_(m_weights, start, end, gen);
};

template void Linear::normal_(float, float, Generator &);
template void Linear::normal_(double, double, Generator &);

template void Linear::uniform_(float, float, Generator &);
template void Linear::uniform_(double, double, Generator &);

// naive implementation
Tensor Linear::forward(const Tensor &x) {

  if (!x.is_contiguous())
    throw std::runtime_error("Input must be contiguous.");
  if (x.dtype() != m_weights.dtype())
    throw std::runtime_error("Input must have the same type as weight matrix.");
  if (x.shape().back() != in_features)
    throw std::runtime_error("Dimension doesn't match");

  std::vector<std::size_t> shape = x.shape();
  std::vector<std::size_t> strides = x.strides();

  std::vector<std::size_t> new_shape(shape);
  new_shape.back() = out_features;
  Tensor out = Tensor(new_shape, x.dtype());

  std::size_t batch_numel{1};

  for (std::size_t k = 0; k < shape.size() - 1; k++)
    batch_numel *= shape[k];

  for (std::size_t b = 0; b < batch_numel; b++) {

    DISPATCH_FLOAT(x.dtype(), scalar_t, {
      const scalar_t *pW = m_weights.data_ptr<scalar_t>();
      const scalar_t *px = x.data_ptr<scalar_t>() + b * in_features;
      scalar_t *py = out.data_ptr<scalar_t>() + b * out_features;
      gemv_kernel(pW, px, py, out_features, in_features);
    })

    if (has_bias()) {
      DISPATCH_FLOAT(x.dtype(), scalar_t, {
        const scalar_t *pb = m_bias.data_ptr<scalar_t>();
        scalar_t *py = out.data_ptr<scalar_t>() + b * out_features;
        vadd_kernel(pb, py, out_features);
      })
    }
  }

  return out;
}