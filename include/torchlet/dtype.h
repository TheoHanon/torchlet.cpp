#pragma once
#include <cstdint>

enum class Dtype { Float32, Float64, Int32, Int64, UInt8, UInt32, UInt64 };

template <typename T> struct CPPTypeToDType;

template <> struct CPPTypeToDType<float> {
  static constexpr Dtype dtype = Dtype::Float32;
};
template <> struct CPPTypeToDType<double> {
  static constexpr Dtype dtype = Dtype::Float64;
};
template <> struct CPPTypeToDType<std::int32_t> {
  static constexpr Dtype dtype = Dtype::Int32;
};
template <> struct CPPTypeToDType<std::int64_t> {
  static constexpr Dtype dtype = Dtype::Int64;
};
template <> struct CPPTypeToDType<std::uint8_t> {
  static constexpr Dtype dtype = Dtype::UInt8;
};
template <> struct CPPTypeToDType<std::uint32_t> {
  static constexpr Dtype dtype = Dtype::UInt32;
};
template <> struct CPPTypeToDType<std::uint64_t> {
  static constexpr Dtype dtype = Dtype::UInt64;
};

#define DISPATCH_FLOAT(dtype, NAME, BODY)                                      \
  switch (dtype) {                                                             \
  case Dtype::Float32: {                                                       \
    using NAME = float;                                                        \
    BODY                                                                       \
  } break;                                                                     \
  case Dtype::Float64: {                                                       \
    using NAME = double;                                                       \
    BODY                                                                       \
  } break;                                                                     \
    throw std::runtime_error("Unsupported dtype");                             \
  }

#define DISPATCH_ALL(dtype, NAME, BODY)                                        \
  switch (dtype) {                                                             \
  case Dtype::Float32: {                                                       \
    using NAME = float;                                                        \
    BODY                                                                       \
  } break;                                                                     \
  case Dtype::Float64: {                                                       \
    using NAME = double;                                                       \
    BODY                                                                       \
  } break;                                                                     \
  case Dtype::Int32: {                                                         \
    using NAME = std::int32_t;                                                 \
    BODY                                                                       \
  } break;                                                                     \
  case Dtype::Int64: {                                                         \
    using NAME = std::int64_t;                                                 \
    BODY                                                                       \
  } break;                                                                     \
  case Dtype::UInt64: {                                                        \
    using NAME = std::uint64_t;                                                \
    BODY                                                                       \
  } break;                                                                     \
  case Dtype::UInt32: {                                                        \
    using NAME = std::uint32_t;                                                \
    BODY                                                                       \
  } break;                                                                     \
  case Dtype::UInt8: {                                                         \
    using NAME = std::uint8_t;                                                 \
    BODY                                                                       \
  } break;                                                                     \
    throw std::runtime_error("Unsupported dtype");                             \
  }
