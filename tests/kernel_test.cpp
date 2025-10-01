#include <gtest/gtest.h>
#include <random>

#include "utils/utils.h"
#include <torchlet/ops/kernel.h>
#include <torchlet/torchlet.h>

template <typename T> class KernelTypedTest : public ::testing::Test {};
using MyTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(KernelTypedTest, MyTypes);

TYPED_TEST(KernelTypedTest, VaddLarge) {
  using T = TypeParam;
  static_assert(std::is_floating_point_v<T>, "This test expects float/double");

  const std::size_t m = 1u << 20;
  std::vector<T> a(m, T{1}), b(m, T{1}), expected(m, T{2});

  vadd_kernel(a.data(), b.data(), m);
  expect_array_equal(b.data(), expected.data(), m);
};

TYPED_TEST(KernelTypedTest, MvBasic) {
  using T = TypeParam;
  std::size_t m = 1 << 9;
  std::size_t n = 1 << 8;

  std::vector<T> W(m * n, T{1.0}), x(m, T{1.0}), y(n, T{0.0}),
      expected(n, static_cast<T>(m));

  gemv_kernel(W.data(), x.data(), y.data(), m, n);
  expect_array_equal(y.data(), expected.data(), n);
};

TYPED_TEST(KernelTypedTest, MvIdentityRandom) {

  using T = TypeParam;
  std::mt19937 engine{42};
  std::uniform_real_distribution<T> dist{T{0}, T{1.0}};

  std::size_t m = 1 << 9;

  std::vector<T> W(m * m, T{0.0}), x(m, T{1.0}), y(m, T{0.0});

  for (auto k = 0; k < m; k++) {
    x[k] = dist(engine);
  }

  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < m; j++) {
      if (i == j)
        W[i * m + j] = 1.0f;
    }
  }
  std::vector<T> expected(x);

  gemv_kernel(W.data(), x.data(), y.data(), m, m);
  expect_array_equal(y.data(), expected.data(), m);
};

TYPED_TEST(KernelTypedTest, VaddSingleElement) {
  using T = TypeParam;
  T a = T{3}, b = T{4};
  vadd_kernel(&a, &b, 1u);
  expect_equal(T{7}, b);
};

TYPED_TEST(KernelTypedTest, VaddWithNegatives) {
  using T = TypeParam;
  std::vector<T> a = {T{1.0}, T{-2.0}, T{3.5}};
  std::vector<T> b = {T{0.5}, T{2.0}, T{-1.5}};
  std::vector<T> expected = {T{1.5}, T{0.0}, T{2.0}};

  vadd_kernel(a.data(), b.data(), a.size());
  expect_array_equal(b.data(), expected.data(), a.size());
};
