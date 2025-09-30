#include <gtest/gtest.h>
#include <torchlet/torchlet.h>

#include "utils/utils.h"

template <typename T> class InitTypedTest : public ::testing::Test {};
using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(InitTypedTest, MyTypes);

TYPED_TEST(InitTypedTest, Reproducable) {

  using T = TypeParam;
  const auto dt = CPPTypeToDType<T>::dtype;
  auto g1 = Generator(10u);
  auto g2 = Generator(10u);

  std::size_t len = 10;

  Tensor t({len}, dt);
  Tensor v({len}, dt);

  torchlet::init::normal_(t, T{0}, T{1}, g1);
  torchlet::init::normal_(v, T{0}, T{1}, g2);
  expect_array_equal(t.data_ptr<T>(), v.data_ptr<T>(), len);

  torchlet::init::uniform_(t, T{0}, T{1}, g1);
  torchlet::init::uniform_(v, T{0}, T{1}, g2);
  expect_array_equal(t.data_ptr<T>(), v.data_ptr<T>(), len);
};