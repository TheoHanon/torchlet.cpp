#pragma once
#include <stdexcept>
#include <torchlet/core/tensor.h>
#include <vector>

#ifndef TL_DEBUG
#if !defined(NDEBUG)
#define TL_DEBUG 1
#else
#define TL_DEBUG 0
#endif
#endif

#define TL_CHECK(COND, MSG)                                                    \
  do {                                                                         \
    if (TL_DEBUG && !(COND))                                                   \
      throw std::runtime_error(MSG);                                           \
  } while (0)

namespace torchlet::detail {

inline void validate_shape(std::vector<size_t> old_shape,
                           std::vector<size_t> new_shape) {

  size_t new_prod = 1;
  size_t old_prod = 1;

  for (size_t k = 0; k < new_shape.size(); k++)
    new_prod *= new_shape[k];
  for (size_t k = 0; k < old_shape.size(); k++)
    old_prod *= old_shape[k];

  if (old_prod != new_prod)
    throw std::invalid_argument("Shapes doesn't match.");
};

inline void validate_contiguous(const std::vector<size_t> &shape,
                                const std::vector<size_t> &strides) {

  for (size_t i = 1; i < strides.size(); i++) {
    if (strides[i] * shape[i] != strides[i - 1])
      throw std::runtime_error("Memory layout is not row major.");
  }
};

inline void check_contiguous(const torchlet::core::Tensor &t,
                             const char *name) {
  TL_CHECK(t.is_contiguous(), std::string(name) + " must be contiguous.");
};

inline void check_same_dtype(const torchlet::core::Tensor &a,
                             const torchlet::core::Tensor &b, const char *an,
                             const char *bn) {
  TL_CHECK(a.dtype() == b.dtype(),
           std::string(an) + " and " + bn + " must have same dtype.");
};

inline void check_rank(const torchlet::core::Tensor &t, int r,
                       const char *name) {
  TL_CHECK((int)t.shape().size() == r,
           std::string(name) + " must be " + std::to_string(r) + "D.");
};

inline void check_rank_ge(const torchlet::core::Tensor &t, int r,
                          const char *name) {
  TL_CHECK((int)t.shape().size() >= r,
           std::string(name) + " must be at least " + std::to_string(r) + "D.");
};

inline void check_dim_eq(const torchlet::core::Tensor &t, int axis,
                         std::size_t v, const char *name, const char *what) {
  TL_CHECK(t.shape().at(axis) == v,
           std::string(name) + " " + what + " mismatch.");
};

inline bool has_data(const torchlet::core::Tensor &t) {
  return t.storage_ptr() != nullptr;
};

} // namespace torchlet::detail