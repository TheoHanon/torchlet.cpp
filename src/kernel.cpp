#include <cassert>
#include <torchlet/kernel.h>

template <typename T>
void gemv_kernel(const T *W, const T *x, T *y, std::size_t m,
                 std::size_t n) noexcept {
  for (auto k = 0; k < m; k++) {
    T tmp(0);
    const T *wrow = W + k * n;
    for (auto i = 0; i < n; i++) {
      tmp += wrow[i] * x[i];
    }
    y[k] = tmp;
  }
}

template <typename T>
void vadd_kernel(const T *x, T *y, std::size_t m) noexcept {

  for (auto k = 0; k < m; k++) {
    y[k] += x[k];
  }
};

template void gemv_kernel(const float *W, const float *x, float *y,
                          std::size_t m, std::size_t n);
template void gemv_kernel(const double *W, const double *x, double *y,
                          std::size_t m, std::size_t n);

template void vadd_kernel(const float *x, float *y, std::size_t m);
template void vadd_kernel(const double *x, double *y, std::size_t m);