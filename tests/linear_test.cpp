#include "utils/utils.h"
#include <gtest/gtest.h>
#include <torchlet/torchlet.h>

using torchlet::core::Tensor, torchlet::module::Linear, torchlet::core::Dtype;

template <typename T> class LinearTypedTest : public ::testing::Test {};
using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinearTypedTest, MyTypes);

TYPED_TEST(LinearTypedTest, ConstructWithCorrectShapes) {

  using T = TypeParam;
  const size_t in = 3, out = 5;
  const auto dt = CPPTypeToDType<T>::dtype;

  Linear lin(in, out, false, dt);

  Tensor x({in}, dt);
  torchlet::ops::init::uniform_(lin.weights(), T{0}, T{1});

  Tensor y = lin.forward(x);

  EXPECT_EQ(y.shape(), (std::vector<size_t>{out}));
  EXPECT_EQ(y.dtype(), dt);
};

TYPED_TEST(LinearTypedTest, ForwardOnesNoBias) {

  using T = TypeParam;
  const size_t in = 3, out = 5;
  const auto dt = CPPTypeToDType<T>::dtype;

  Linear lin(in, out, false, dt);
  EXPECT_THROW(lin.bias(), std::runtime_error);

  Tensor x = Tensor::ones({in}, dt);
  torchlet::ops::init::uniform_(lin.weights(), T{1},
                                T{1}); // work around to set to one.

  Tensor z = lin.forward(x);
  expect_array_equal(z.data_ptr<T>(), std::vector<T>(out, T{in}).data(), out);
};

TYPED_TEST(LinearTypedTest, ForwardOnesBias) {

  using T = TypeParam;
  const size_t in = 3, out = 5;
  const auto dt = CPPTypeToDType<T>::dtype;

  Linear lin(in, out, true, dt);

  Tensor x = Tensor::ones({in}, dt);
  torchlet::ops::init::uniform_(lin.weights(), T{1},
                                T{1}); // work around to set to one.
  torchlet::ops::init::uniform_(lin.bias(), T{1}, T{1});

  Tensor z = lin.forward(x);
  expect_array_equal(z.data_ptr<T>(), std::vector<T>(out, T{in + 1}).data(),
                     out);
};

TYPED_TEST(LinearTypedTest, ForwardBatch2DOneNoBias) {
  using T = TypeParam;
  const std::size_t B = 3, in = 4, out = 2;
  const auto dt = CPPTypeToDType<T>::dtype;

  Linear lin(in, out, false, dt);

  torchlet::ops::init::uniform_(lin.weights(), T{1}, T{1}); // W = ones

  // x shape [B, in]: rows = [1..in], [2..2*in step 2], [3..3*in step 3]
  Tensor x = Tensor::zeros({B, in}, dt);
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t i = 0; i < in; ++i) {
      x.assign_({b, i}, static_cast<T>((b + 1) * (i + 1))); // simple pattern
    }
  }

  std::vector<T> row_sum(B, T{0});
  for (std::size_t b = 0; b < B; ++b)
    for (std::size_t i = 0; i < in; ++i)
      row_sum[b] += static_cast<T>((b + 1) * (i + 1));

  Tensor y = lin.forward(x);
  ASSERT_EQ(y.shape(), (std::vector<std::size_t>{B, out}));

  const T *py = y.data_ptr<T>();
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t j = 0; j < out; ++j) {
      const std::size_t idx = b * out + j;
      EXPECT_EQ(py[idx], row_sum[b]) << "b=" << b << " j=" << j;
    }
  }
};

TYPED_TEST(LinearTypedTest, ForwardBatch2DOneBias) {
  using T = TypeParam;
  const std::size_t B = 3, in = 4, out = 2;
  const auto dt = CPPTypeToDType<T>::dtype;

  Linear lin(in, out, true, dt);

  torchlet::ops::init::uniform_(lin.weights(), T{1}, T{1}); // W = ones
  torchlet::ops::init::uniform_(lin.bias(), T{1}, T{1});    // need to change

  // x shape [B, in]: rows = [1..in], [2..2*in step 2], [3..3*in step 3]
  Tensor x = Tensor::zeros({B, in}, dt);
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t i = 0; i < in; ++i) {
      x.assign_({b, i}, static_cast<T>((b + 1) * (i + 1))); // simple pattern
    }
  }

  std::vector<T> row_sum(B, T{1});
  for (std::size_t b = 0; b < B; ++b)
    for (std::size_t i = 0; i < in; ++i)
      row_sum[b] += static_cast<T>((b + 1) * (i + 1));

  Tensor y = lin.forward(x);
  ASSERT_EQ(y.shape(), (std::vector<std::size_t>{B, out}));

  const T *py = y.data_ptr<T>();
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t j = 0; j < out; ++j) {
      const std::size_t idx = b * out + j;
      EXPECT_EQ(py[idx], row_sum[b]) << "b=" << b << " j=" << j;
    }
  }
};

TYPED_TEST(LinearTypedTest, ForwardBatch3DOnesNoBias) {
  using T = TypeParam;
  const std::size_t B1 = 2, B2 = 3, in = 5, out = 4;
  const auto dt = CPPTypeToDType<T>::dtype;

  Linear lin(in, out, false, dt);

  torchlet::ops::init::uniform_(lin.weights(), T{1}, T{1});

  Tensor x = Tensor::zeros({B1, B2, in}, dt);

  for (std::size_t b1 = 0; b1 < B1; ++b1)
    for (std::size_t b2 = 0; b2 < B2; ++b2)
      for (std::size_t i = 0; i < in; ++i)
        x.assign_({b1, b2, i},
                  static_cast<T>((b1 + 1) + 10 * (b2 + 1) + 20 * (i + 1)));

  auto sum_row = [&](std::size_t b1, std::size_t b2) {
    T s = T{0};
    for (std::size_t i = 0; i < in; ++i)
      s += static_cast<T>((b1 + 1) + 10 * (b2 + 1) + 20 * (i + 1));
    return s;
  };

  Tensor y = lin.forward(x);
  ASSERT_EQ(y.shape(), (std::vector<std::size_t>{B1, B2, out}));

  const T *py = y.data_ptr<T>();
  for (std::size_t b1 = 0; b1 < B1; ++b1)
    for (std::size_t b2 = 0; b2 < B2; ++b2) {
      T s = sum_row(b1, b2);
      for (std::size_t j = 0; j < out; ++j) {
        const std::size_t idx = ((b1 * B2) + b2) * out + j;
        EXPECT_EQ(py[idx], s) << "b1=" << b1 << " b2=" << b2 << " j=" << j;
      }
    }
};

TYPED_TEST(LinearTypedTest, ForwardBatch3DOnesBias) {
  using T = TypeParam;
  const std::size_t B1 = 2, B2 = 3, in = 5, out = 4;
  const auto dt = CPPTypeToDType<T>::dtype;

  Linear lin(in, out, true, dt);

  torchlet::ops::init::uniform_(lin.weights(), T{1}, T{1});
  torchlet::ops::init::uniform_(lin.bias(), T{2}, T{2});

  Tensor x = Tensor::zeros({B1, B2, in}, dt);

  for (std::size_t b1 = 0; b1 < B1; ++b1)
    for (std::size_t b2 = 0; b2 < B2; ++b2)
      for (std::size_t i = 0; i < in; ++i)
        x.assign_({b1, b2, i},
                  static_cast<T>((b1 + 1) + 10 * (b2 + 1) + 20 * (i + 1)));

  auto sum_row = [&](std::size_t b1, std::size_t b2) {
    T s = T{2};
    for (std::size_t i = 0; i < in; ++i)
      s += static_cast<T>((b1 + 1) + 10 * (b2 + 1) + 20 * (i + 1));
    return s;
  };

  Tensor y = lin.forward(x);
  ASSERT_EQ(y.shape(), (std::vector<std::size_t>{B1, B2, out}));

  const T *py = y.data_ptr<T>();
  for (std::size_t b1 = 0; b1 < B1; ++b1)
    for (std::size_t b2 = 0; b2 < B2; ++b2) {
      T s = sum_row(b1, b2);
      for (std::size_t j = 0; j < out; ++j) {
        const std::size_t idx = ((b1 * B2) + b2) * out + j;
        EXPECT_EQ(py[idx], s) << "b1=" << b1 << " b2=" << b2 << " j=" << j;
      }
    }
};

TYPED_TEST(LinearTypedTest, ForwardTypeMismatch) {

  using T = TypeParam;
  const std::size_t B = 3, in = 5, out = 3;
  const auto dt = CPPTypeToDType<T>::dtype;

  Linear lin(in, out, false, CPPTypeToDType<T>::dtype);

  const auto wrong_dt =
      (dt == Dtype::Float32) ? Dtype::Float64 : Dtype::Float32;

  Tensor x({in}, wrong_dt);
  EXPECT_THROW(lin.forward(x), std::runtime_error);
};

TEST(LinearTest, ForwardDimMismatch) {
  const auto dt = Dtype::Float32;
  Linear lin(5, 3, false, dt);

  Tensor x = Tensor::zeros({7}, dt);
  torchlet::ops::init::uniform_(lin.weights(), 0.0f, 0.0f);

  EXPECT_THROW(lin.forward(x), std::runtime_error);
};

TEST(LinearTest, ForwardNonContiguous) {
  const auto dt = Dtype::Float32;
  Linear lin(3, 2, false, dt);

  Tensor base = Tensor::ones({2, 3}, dt);
  Tensor x = base.permute(0, 1);

  torchlet::ops::init::uniform_(lin.weights(), 0.0f, 0.0f);

  EXPECT_THROW(lin.forward(x), std::runtime_error);
};