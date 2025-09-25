#pragma once
#include <cstdint>

enum class Dtype {
    Float32,
    Float64, 
    Int32, 
    Int64, 
    UInt8,
};


template <typename T>
struct DTypeToCPPType;

template <> struct DTypeToCPPType<float> {static constexpr Dtype  dtype = Dtype::Float32;};
template <> struct DTypeToCPPType<double> {static constexpr Dtype dtype = Dtype::Float64;};
template <> struct DTypeToCPPType<std::int32_t> {static constexpr Dtype dtype = Dtype::Int32;};
template <> struct DTypeToCPPType<std::int64_t> {static constexpr Dtype dtype = Dtype::Int64;};
template <> struct DTypeToCPPType<std::uint8_t> {static constexpr Dtype dtype = Dtype::UInt8;};


template <Dtype>
struct CPPTypeToDtype;

template <> struct CPPTypeToDtype<Dtype::Float32> {using dtype = float;};
template <> struct CPPTypeToDtype<Dtype::Float64> {using dtype = double;};
template <> struct CPPTypeToDtype<Dtype::Int32> {using dtype = int32_t;};
template <> struct CPPTypeToDtype<Dtype::Int64> {using dtype = int64_t;};
template <> struct CPPTypeToDtype<Dtype::UInt8> {using dtype = uint8_t;};


