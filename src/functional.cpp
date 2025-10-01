#include <cmath>
#include <torchlet/ops/functional.h>
#include <torchlet/ops/kernel.h>

using torchlet::core::Tensor;

Tensor torchlet::ops::linear(const Tensor &x, const Tensor &weights,
                             const Tensor &bias) {

  if (!x.is_contiguous())
    throw std::runtime_error("Input must be contiguous.");
  if (x.dtype() != weights.dtype())
    throw std::runtime_error("Input must have the same type as weight matrix.");
  if (x.shape().back() != weights.shape().back())
    throw std::runtime_error("Dimension doesn't match");

  std::vector<std::size_t> x_shape = x.shape(), w_shape = weights.shape();
  std::size_t in_features = x_shape.back(), out_features = w_shape.front();

  std::vector<std::size_t> new_shape(x_shape);
  new_shape.back() = out_features;
  Tensor out(new_shape, x.dtype());

  std::size_t batch_numel{1};

  for (std::size_t k = 0; k < x_shape.size() - 1; k++)
    batch_numel *= x_shape[k];

  for (std::size_t b = 0; b < batch_numel; b++) {

    DISPATCH_FLOAT(x.dtype(), scalar_t, {
      const scalar_t *pW = weights.data_ptr<scalar_t>();
      const scalar_t *px = x.data_ptr<scalar_t>() + b * in_features;
      scalar_t *py = out.data_ptr<scalar_t>() + b * out_features;
      gemv_kernel(pW, px, py, out_features, in_features);
    })

    if (bias.storage_ptr() != nullptr) {
      DISPATCH_FLOAT(x.dtype(), scalar_t, {
        const scalar_t *pb = bias.data_ptr<scalar_t>();
        scalar_t *py = out.data_ptr<scalar_t>() + b * out_features;
        vadd_kernel(pb, py, out_features);
      })
    }
  }

  return out;
};

Tensor torchlet::ops::gelu(const Tensor &x) {
  if (!x.is_contiguous())
    throw std::runtime_error("GELU: non-contiguous not implemented.");

  Tensor out(x.shape(), x.dtype());

  DISPATCH_FLOAT(x.dtype(), scalar_t, {
    const scalar_t *px = x.data_ptr<scalar_t>();
    scalar_t *po = out.data_ptr<scalar_t>();
    const std::size_t N = x.numel();

    constexpr scalar_t half = scalar_t{0.5};
    constexpr scalar_t coeff = scalar_t{0.044715};
    const scalar_t sqrt_2_pi =
        std::sqrt(scalar_t{0.63661977236758134308}); // sqrt(2/pi)

    for (auto k = 0; k < x.numel(); ++k) {
      scalar_t val = px[k];
      scalar_t x3 = val * val * val;

      po[k] = half * val *
              (scalar_t{1} + std::tanh(sqrt_2_pi * std::fma(coeff, x3, val)));
    }
  })

  return out;
};