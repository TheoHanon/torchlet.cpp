#pragma once

#include <cstdlib>

/// @brief General matrix vector product kernel
/// @tparam T type
/// @param W m x n matrix
/// @param x n input vector
/// @param y m output vector
/// @param m rowsize
/// @param n colsize
template <typename T>
void gemv_kernel(const T *W, const T *x, T *y, std::size_t m,
                 std::size_t n) noexcept;

/// @brief Vector addition
/// @tparam T type
/// @param x m-dim vector to add
/// @param y m-dim vector modified in-place
/// @param m vector size
template <typename T>
void vadd_kernel(const T *x, T *y, std::size_t m) noexcept;