#include <gtest/gtest.h>
#include <random>
#include <torchlet/torchlet.h>

TEST(Kernel, VaddLarge) {

  std::size_t m = 1 << 20;
  std::vector<float> a(m, 1.0f), b(m, 1.0f), ans(m, 2.0f);

  vadd_kernel(a.data(), b.data(), m);

  for (auto k = 0; k < m; k++) {
    EXPECT_FLOAT_EQ(ans[k], b[k]) << "mismatch at k=" << k;
  }
};

TEST(Kernel, MvBasic) {
  std::size_t m = 1 << 9;
  std::size_t n = 1 << 8;

  std::vector<float> W(m * n, 1.0f), x(m, 1.0f), y(n, 0.0f),
      ans(n, static_cast<float>(m));

  gemv_kernel(W.data(), x.data(), y.data(), m, n);

  for (auto k = 0; k < n; k++) {
    EXPECT_FLOAT_EQ(ans[k], y[k]) << "mismatch at k=" << k;
  }
};

TEST(Kernel, MvIdentityRandom) {

  std::mt19937 engine{42};
  std::uniform_real_distribution<float> dist{0.0f, 1.0f};

  std::size_t m = 1 << 9;

  std::vector<float> W(m * m, 0.0f), x(m, 1.0f), y(m, 0.0f);

  for (auto k = 0; k < m; k++) {
    x[k] = dist(engine);
  }

  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < m; j++) {
      if (i == j)
        W[i * m + j] = 1.0f;
    }
  }
  std::vector<float> ans(x);

  gemv_kernel(W.data(), x.data(), y.data(), m, m);

  for (auto k = 0; k < m; k++) {
    EXPECT_FLOAT_EQ(ans[k], y[k]) << "mismatch at k=" << k;
  }
};

TEST(Kernel, VaddSingleElement) {
  float a = 3.0f, b = 4.0f;
  vadd_kernel(&a, &b, 1);
  EXPECT_FLOAT_EQ(b, 7.0f);
};

TEST(Kernel, VaddWithNegatives) {
  std::vector<float> a = {1.0f, -2.0f, 3.5f};
  std::vector<float> b = {0.5f, 2.0f, -1.5f};
  std::vector<float> ans = {1.5f, 0.0f, 2.0f};

  vadd_kernel(a.data(), b.data(), a.size());

  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_FLOAT_EQ(b[i], ans[i]) << "i=" << i;
};
