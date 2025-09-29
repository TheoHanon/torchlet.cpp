#include <gtest/gtest.h>
#include <torchlet/torchlet.h>

template <typename T> class TensorTypedTest : public ::testing::Test {};
using MyTypes = ::testing::Types<float, double>;
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

  for (auto k = 0; k < z.numel(); ++k) {
    if constexpr (std::is_same_v<T, float>) {
      EXPECT_FLOAT_EQ(zp[k], 0.0f);
    } else {
      EXPECT_DOUBLE_EQ(zp[k], 0.0);
    }
  }

  Tensor o = Tensor::ones({5}, dt);
  EXPECT_EQ(o.shape(), (std::vector<size_t>{5}));
  EXPECT_EQ(o.numel(), 5u);
  EXPECT_TRUE(o.is_contiguous());
  ASSERT_NE(o.storage_ptr(), nullptr);

  const T *op = o.data_ptr<T>();
  for (auto k = 0; k < o.numel(); ++k) {
    if constexpr (std::is_same_v<T, float>) {
      EXPECT_FLOAT_EQ(op[k], 1.0f);
    } else {
      EXPECT_DOUBLE_EQ(op[k], 1.0);
    }
  }
};

TYPED_TEST(TensorTypedTest, FillAndAssignAndItem) {
  using T = TypeParam;
  auto dt = CPPTypeToDType<T>::dtype;

  Tensor t = Tensor::zeros({3, 2}, dt);
  t.fill_(T{2.5});

  for (size_t i = 0; i < t.numel(); ++i) {
    if constexpr (std::is_same_v<T, float>) {
      EXPECT_FLOAT_EQ(t.data_ptr<T>()[i], 2.5f);
    } else {
      EXPECT_DOUBLE_EQ(t.data_ptr<T>()[i], 2.5);
    }
  }

  t.assign_({1, 0}, T{7});

  if constexpr (std::is_same_v<T, float>) {
    EXPECT_FLOAT_EQ(t.index({1, 0}).item<float>(), 7.0f);
    EXPECT_FLOAT_EQ(t.index({0, 1}).item<float>(), 2.5f);
  } else {
    EXPECT_DOUBLE_EQ(t.index({1, 0}).item<double>(), 7.0);
    EXPECT_DOUBLE_EQ(t.index({0, 1}).item<double>(), 2.5);
  }
};
