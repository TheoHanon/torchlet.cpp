#include <gtest/gtest.h>
#include <limits>

#include "utils/utils.h"
#include <torchlet/torchlet.h>

using torchlet::core::Tensor, torchlet::core::Generator, torchlet::core::Dtype;

template <typename T> class TensorTypedTest : public ::testing::Test {};
using MyTypes = ::testing::Types<float, double, int32_t, uint32_t, uint8_t,
                                 uint32_t, uint64_t>;
TYPED_TEST_SUITE(TensorTypedTest, MyTypes);

TYPED_TEST(TensorTypedTest, ZerosOnesBasicProps) {

  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;
  Tensor z = Tensor::zeros({2, 3, 4}, dt);

  EXPECT_EQ(z.shape(), (std::vector<size_t>{2, 3, 4}));
  EXPECT_EQ(z.numel(), 24u);
  EXPECT_TRUE(z.is_contiguous());
  ASSERT_NE(z.storage_ptr(), nullptr);

  const T *zp = z.data_ptr<T>();
  expect_array_equal(zp, std::vector<T>(z.numel(), T{0}).data(), z.numel());

  Tensor o = Tensor::ones({5}, dt);
  EXPECT_EQ(o.shape(), (std::vector<size_t>{5}));
  EXPECT_EQ(o.numel(), 5u);
  EXPECT_TRUE(o.is_contiguous());
  ASSERT_NE(o.storage_ptr(), nullptr);

  const T *op = o.data_ptr<T>();
  expect_array_equal(op, std::vector<T>(o.numel(), T{1}).data(), o.numel());
};

TYPED_TEST(TensorTypedTest, FillAndAssignAndItem) {
  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;

  Tensor t = Tensor::zeros({3, 2}, dt);
  t.fill_(T{2});
  expect_array_equal(t.data_ptr<T>(), std::vector<T>(t.numel(), T{2}).data(),
                     t.numel());

  t.assign_({1, 0}, T{7});
  expect_equal(t.index({1, 0}).item<T>(), T{7});
  expect_equal(t.index({0, 1}).item<T>(), T{2});
};

TYPED_TEST(TensorTypedTest, PointIndexSharesStorageAndOffset) {
  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;

  Tensor t = Tensor::zeros({4, 3}, dt);
  t.fill_(T{1});

  Tensor v = t.index({2, 1});
  EXPECT_EQ(v.storage_ptr(), t.storage_ptr());
  EXPECT_FALSE(v.shape().empty());

  v.fill_(T{9});
  expect_equal(v.item<T>(), t.data_ptr<T>()[v.elem_offset()]);
};

TYPED_TEST(TensorTypedTest, SliceIndex) {
  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;

  Tensor t = Tensor::zeros({3, 4}, dt);
  Tensor row1 = t.index(
      {torchlet::core::index::Slice(1), torchlet::core::index::Slice(0, 4)});

  EXPECT_EQ(row1.shape(), (std::vector<size_t>{4}));
  EXPECT_EQ(row1.storage_ptr(), t.storage_ptr());
  EXPECT_FALSE(row1.is_contiguous());

  row1.fill_(T{5});
  for (size_t i = 0; i < t.shape().back(); i++) {
    expect_equal(t.index({1, i}).item<T>(), T{5});
  }
};

TYPED_TEST(TensorTypedTest, PermuteDims) {
  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;

  Tensor t = Tensor::ones({2, 3, 4}, dt);
  Tensor p = t.permute(0, 2);

  EXPECT_EQ(p.shape(), (std::vector<size_t>({4, 3, 2})));
  EXPECT_EQ(p.storage_ptr(), t.storage_ptr());
  EXPECT_FALSE(p.is_contiguous());
};

TYPED_TEST(TensorTypedTest, View) {

  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;

  Tensor t = Tensor::ones({2, 3, 4}, dt);
  Tensor v = t.view({6, 4});

  EXPECT_EQ(v.shape(), (std::vector<size_t>{6, 4}));
  EXPECT_EQ(v.numel(), 24u);
  EXPECT_EQ(v.storage_ptr(), t.storage_ptr());
  EXPECT_TRUE(v.is_contiguous());
};

TYPED_TEST(TensorTypedTest, ViewNonContiguous) {

  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;

  Tensor t = Tensor::ones({2, 3, 4}, dt);
  Tensor p = t.permute(0, 1);

  EXPECT_THROW(p.view({6, 4}), std::runtime_error);
};

TYPED_TEST(TensorTypedTest, ElemOffset) {
  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;

  Tensor t = Tensor::zeros({4, 5}, dt);

  Tensor row2 = t.index(
      {torchlet::core::index::Slice(2), torchlet::core::index::Slice(0, 5)});
  EXPECT_EQ(row2.storage_ptr(), t.storage_ptr());
  EXPECT_EQ(row2.elem_offset(), 10);

  row2.assign_({0}, T{3});
  expect_equal(t.index({2, 0}).item<T>(), T{3});
};

TEST(TensorTest, ContiguityInvariants) {
  auto dt = Dtype::Float32;

  Tensor t = Tensor::zeros({2, 3, 4}, dt);
  EXPECT_TRUE(t.is_contiguous());
  Tensor p = t.permute(0, 2);
  EXPECT_FALSE(p.is_contiguous());

  Tensor s = p.index({torchlet::core::index::Slice(0, 4),
                      torchlet::core::index::Slice(0, 3),
                      torchlet::core::index::Slice(0, 2)});
  EXPECT_FALSE(s.is_contiguous());
}

TEST(TensorTest, IndexOutOfBounds) {
  auto dt = Dtype::Float32;
  Tensor t = Tensor::zeros({3, 3}, dt);
  EXPECT_THROW(t.index({3, 0}), std::invalid_argument);
}

TEST(TensorTest, ViewMismatchedNumel) {
  auto dt = Dtype::Float32;
  Tensor t = Tensor::zeros({2, 3}, dt);
  EXPECT_THROW(t.view({4, 2}), std::invalid_argument);
}

class SizeParam : public ::testing::TestWithParam<std::vector<size_t>> {};
INSTANTIATE_TEST_SUITE_P(TensorSizeTest, SizeParam,
                         ::testing::Values(std::vector<size_t>{0},
                                           std::vector<size_t>{1},
                                           std::vector<size_t>{7},
                                           std::vector<size_t>{2, 3},
                                           std::vector<size_t>{3, 4, 5}));

TEST_P(SizeParam, ZerosNumelAndFill) {

  auto dt = Dtype::Float32;
  const auto shape = GetParam();

  size_t numel = 1;
  for (const auto &s : shape)
    numel *= s;

  Tensor t(shape, dt);
  EXPECT_EQ(numel, t.numel());

  t.fill_(5.0f);
  const auto *ptr = t.data_ptr<float>();

  expect_array_equal(ptr, std::vector<float>(t.numel(), 5.0f).data(),
                     t.numel());
};