#include <cstddef>
#include <gtest/gtest.h>

template <typename T> inline constexpr T tol_for() {
  if constexpr (std::is_same_v<T, float>)
    return T(8) * std::numeric_limits<T>::epsilon();
  if constexpr (std::is_same_v<T, double>)
    return T(8) * std::numeric_limits<T>::epsilon();
  return T(0);
};

template <typename T>
inline void expect_equal(const T &a, const T &b, const char *msg = nullptr) {
  if constexpr (std::is_floating_point<T>()) {
    EXPECT_NEAR(a, b, tol_for<T>()) << (msg ? msg : "");
  } else {
    EXPECT_EQ(a, b) << (msg ? msg : "");
  }
};

template <typename T>
inline void expect_array_equal(const T *a, const T *b, std::size_t n) {
  for (auto i = 0; i < n; ++i) {
    expect_equal(a[i], b[i], (std::string("i=") + std::to_string(i)).c_str());
  }
};