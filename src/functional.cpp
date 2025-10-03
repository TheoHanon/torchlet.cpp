#include "detail/validators.h"
#include <torchlet/iterator/iterator.h>
#include <torchlet/ops/functional.h>
#include <torchlet/ops/kernel.h>

using torchlet::core::Tensor, torchlet::iterator::ContiguousIterator;

// Tensor scaled_dot_product_attention(const Tensor &Q, const Tensor &K,
//                                     const Tensor &V) {

//   auto k_shape = K.shape(), q_shape = Q.shape();
//   auto qv_shape = q_shape;
//   qv_shape.back() = k_shape.at(k_shape.size() - 2);

//   Tensor qv(qv_shape, Q.dtype());
//   ContiguousIterator it = ContiguousIterator(&qv, {&Q, &K});
//   std::size_t m =

//   DISPATCH_FLOAT(Q.dtype(), scalar_t, {
//     it.for_each_with_inputs(
//         [&](uint8_t *optr, const uint8_t **iptrs, size_t n_in) {
//           const scalar_t *qp = reinterpret_cast<const scalar_t *>(iptrs[0]);
//           const scalar_t *kp = reinterpret_cast<const scalar_t *>(iptrs[1]);
//           scalar_t *qkp = reinterpret_cast<scalar_t *>(optr);
//         });
//   })
// };

Tensor torchlet::ops::linear(const Tensor &x, const Tensor &weights,
                             const Tensor &bias) {

  torchlet::detail::check_contiguous(x, "x");
  torchlet::detail::check_contiguous(weights, "weights");
  torchlet::detail::check_same_dtype(x, weights, "x", "weights");
  torchlet::detail::check_rank_ge(x, 1, "x");
  torchlet::detail::check_rank(weights, 2, "weights");

  const auto &xs = x.shape();
  const auto &ws = weights.shape();
  const size_t inF = xs.back();
  const size_t outF = ws.front();
  bool has_bias = false;

  torchlet::detail::check_dim_eq(weights, 1, inF, "weights", "in_features");

  if (torchlet::detail::has_data(bias)) {
    has_bias = true;
    torchlet::detail::check_contiguous(bias, "bias");
    torchlet::detail::check_same_dtype(bias, x, "bias", "x");
    torchlet::detail::check_rank(bias, 1, "bias");
    torchlet::detail::check_dim_eq(bias, 0, outF, "bias", "length");
  }

  auto out_shape = xs;
  out_shape.back() = outF;
  Tensor out(out_shape, x.dtype());

  ContiguousIterator it(&out, {&x});

  DISPATCH_FLOAT(x.dtype(), scalar_t, {
    const scalar_t *pW = weights.data_ptr<scalar_t>();
    const scalar_t *pb = has_bias ? bias.data_ptr<scalar_t>() : nullptr;

    it.for_each_with_inputs([&](uint8_t *optr, const uint8_t **iptrs, size_t) {
      const scalar_t *px = reinterpret_cast<const scalar_t *>(iptrs[0]);
      scalar_t *py = reinterpret_cast<scalar_t *>(optr);
      mvb_kernel(pW, px, pb, py, outF, inF);
    });
  })

  return out;
};

Tensor torchlet::ops::gelu(const Tensor &x) {

  torchlet::detail::check_contiguous(x, "x");
  torchlet::detail::check_rank_ge(x, 1, "x");

  Tensor out(x.shape(), x.dtype());
  ContiguousIterator it(&out, {&x});
  std::size_t nfeat = it.input_dim;

  DISPATCH_FLOAT(x.dtype(), scalar_t, {
    it.for_each_with_inputs([&](uint8_t *optr, const uint8_t **iptrs, size_t) {
      const scalar_t *px = reinterpret_cast<const scalar_t *>(iptrs[0]);
      scalar_t *py = reinterpret_cast<scalar_t *>(optr);
      gelu_kernel(px, py, nfeat);
    });
  })
  return out;
};

Tensor torchlet::ops::softmax(const Tensor &x) {
  torchlet::detail::check_contiguous(x, "x");
  torchlet::detail::check_rank_ge(x, 1, "x");

  Tensor out(x.shape(), x.dtype());
  ContiguousIterator it(&out, {&x});
  std::size_t nfeat = it.input_dim;

  DISPATCH_FLOAT(x.dtype(), scalar_t, {
    it.for_each_with_inputs([&](uint8_t *optr, const uint8_t **iptrs, size_t) {
      const scalar_t *px = reinterpret_cast<const scalar_t *>(iptrs[0]);
      scalar_t *py = reinterpret_cast<scalar_t *>(optr);
      softmax_kernel(px, py, nfeat);
    });
  })
  return out;
};

Tensor torchlet::ops::log_softmax(const Tensor &x) {
  torchlet::detail::check_contiguous(x, "x");
  torchlet::detail::check_rank_ge(x, 1, "x");

  Tensor out(x.shape(), x.dtype());
  ContiguousIterator it(&out, {&x});
  std::size_t nfeat = it.input_dim;

  DISPATCH_FLOAT(x.dtype(), scalar_t, {
    it.for_each_with_inputs([&](uint8_t *optr, const uint8_t **iptrs, size_t) {
      const scalar_t *px = reinterpret_cast<const scalar_t *>(iptrs[0]);
      scalar_t *py = reinterpret_cast<scalar_t *>(optr);
      log_softmax_kernel(px, py, nfeat);
    });
  })
  return out;
};
