#include <Accelerate/Accelerate.h>
#include <cassert>
#include <cmath>
#include <torchlet/ops/kernel.h>

template <typename T>
void mm_kernel(const T *A, const T *B, T *C, std::size_t m, std::size_t n,
               std::size_t k) {

  T acc;
  for (std::size_t i = 0; i < m; ++i) {
    const T *Ai = A + i * k;
    for (std::size_t j = 0; j < n; ++j) {
      acc = T{0};
      for (std::size_t l = 0; l < k; ++l) {
        acc = std::fma(Ai[l], B[n * l + j], acc);
      }
      C[i * n + j] = acc;
    }
  }
};

template <typename T>
void mvb_kernel(const T *W, const T *x, const T *b, T *y, std::size_t m,
                std::size_t n) noexcept {

  for (auto k = 0; k < m; k++) {
    T acc = b ? b[k] : T{0};
    const T *wrow = W + k * n;
    for (auto i = 0; i < n; i++) {
      acc += wrow[i] * x[i];
    }
    y[k] = acc;
  }
}

void mvb_blas_kernel(const float *__restrict W, const float *__restrict x,
                     const float *__restrict b, float *__restrict y,
                     std::size_t m, std::size_t n) noexcept {

  float alpha = 1.0f, beta = 0.0f;
  const int incx = 1, incy = 1;

  if (b) {
    beta = 1.0f;
    std::memcpy(y, b, m * sizeof(float));
  }

  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, W, n, x, incx, beta, y,
              incy);
};

template <typename T>
void vadd_kernel(const T *x, T *y, std::size_t m) noexcept {
  for (auto k = 0; k < m; k++) {
    y[k] += x[k];
  }
};

template <typename T>
void gelu_kernel(const T *x, T *y, std::size_t m) noexcept {

  constexpr T half = T{0.5};
  constexpr T coeff = T{0.044715};
  constexpr T sqrt_2_over_pi = T{0.7978845608028654};

  for (auto k = 0; k < m; k++) {
    const T vx = x[k];
    const T x3 = vx * vx * vx;
    const T arg = std::fma(coeff, x3, vx);
    const T t = std::tanh(sqrt_2_over_pi * arg);
    y[k] = half * vx * (T{1} + t);
  };
};

template <typename T>
void softmax_kernel(const T *x, T *y, std::size_t m) noexcept {
  T max = static_cast<T>(std::numeric_limits<T>::lowest());

  for (std::size_t k = 0; k < m; ++k) {
    const T xv = x[k];
    max = (xv > max) ? xv : max;
  }

  T sum = T{0};
  for (std::size_t k = 0; k < m; ++k) {
    const T xv = x[k] - max;
    const T ev = std::exp(xv);
    y[k] = ev;
    sum += ev;
  }

  if (sum == T{0}) {
    const T val = T{1} / static_cast<T>(m);
    for (std::size_t k = 0; k < m; ++k)
      y[k] = val;
    return;
  }

  const T inv_sum = T{1} / sum;
  for (std::size_t k = 0; k < m; ++k) {
    y[k] *= inv_sum;
  }
};

template <typename T>
void log_softmax_kernel(const T *x, T *y, std::size_t m) noexcept {
  T max = static_cast<T>(std::numeric_limits<T>::lowest());

  for (std::size_t k = 0; k < m; ++k) {
    const T xv = x[k];
    max = (xv > max) ? xv : max;
  }
  if (max == -std::numeric_limits<T>::infinity()) {
    for (std::size_t k = 0; k < m; ++k)
      y[k] = -std::numeric_limits<T>::infinity();
    return;
  }

  T sum = T{0};
  for (std::size_t k = 0; k < m; ++k) {
    const T xv = x[k] - max;
    y[k] = xv;
    sum += std::exp(xv);
  }

  if (sum == T{0}) {
    for (std::size_t k = 0; k < m; ++k)
      y[k] = -std::numeric_limits<T>::infinity();
    return;
  }

  T logsum = std::log(sum);
  for (std::size_t k = 0; k < m; ++k) {
    y[k] = y[k] - logsum;
  }
};

template void mm_kernel(const float *A, const float *B, float *C, std::size_t m,
                        std::size_t n, std::size_t k);
template void mm_kernel(const double *A, const double *B, double *C,
                        std::size_t m, std::size_t n, std::size_t k);

template void mvb_kernel(const float *W, const float *x, const float *b,
                         float *y, std::size_t m, std::size_t n);
template void mvb_kernel(const double *W, const double *x, const double *b,
                         double *y, std::size_t m, std::size_t n);

template void vadd_kernel(const float *x, float *y, std::size_t m);
template void vadd_kernel(const double *x, double *y, std::size_t m);

template void gelu_kernel(const float *x, float *y, std::size_t m);
template void gelu_kernel(const double *x, double *y, std::size_t m);

template void softmax_kernel(const float *x, float *y, std::size_t m);
template void softmax_kernel(const double *x, double *y, std::size_t m);

template void log_softmax_kernel(const float *x, float *y, std::size_t m);
template void log_softmax_kernel(const double *x, double *y, std::size_t m);