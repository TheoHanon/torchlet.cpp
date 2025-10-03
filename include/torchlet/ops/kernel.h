#pragma once

#include <cstdlib>

/// @brief Matrix vector product kernel
/// @tparam T double | float
/// @param W m x n matrix
/// @param x n input vector
/// @param b m bias vector
/// @param y m output vector
/// @param m rowsize
/// @param n colsize
template <typename T>
void mvb_kernel(const T *W, const T *x, const T *b, T *y, std::size_t m,
                std::size_t n) noexcept;

void mvb_blas_kernel(const float *__restrict W, const float *__restrict x,
                     const float *__restrict b, float *__restrict y,
                     std::size_t m, std::size_t n) noexcept;

/// @brief Matrix-matrix product kernel
/// @tparam T double | float
/// @param A m x k matrix
/// @param B k x n matrix
/// @param C m x n matrix
/// @param m rowsize A
/// @param n colsize B
/// @param k common dim A, B
template <typename T>
void mm_kernel(const T *A, const T *B, T *C, std::size_t m, std::size_t n,
               std::size_t k);

/// @brief Vector addition
/// @tparam T type
/// @param x m-dim vector to add
/// @param y m-dim vector modified in-place
/// @param m vector size
template <typename T>
void vadd_kernel(const T *x, T *y, std::size_t m) noexcept;

template <typename T>
void gelu_kernel(const T *x, T *y, std::size_t m) noexcept;

template <typename T>
void softmax_kernel(const T *x, T *y, std::size_t m) noexcept;

template <typename T>
void log_softmax_kernel(const T *x, T *y, std::size_t m) noexcept;