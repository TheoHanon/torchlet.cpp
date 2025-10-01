#pragma once
#include <torchlet/core/rng.h>
#include <torchlet/core/tensor.h>

namespace torchlet::ops::init {

template <typename T>
void normal_(
    torchlet::core::Tensor &tensor, T mean, T stdev,
    torchlet::core::Generator &gen = torchlet::core::Generator::global());
template <typename T>
void uniform_(
    torchlet::core::Tensor &tensor, T start, T end,
    torchlet::core::Generator &gen = torchlet::core::Generator::global());

} // namespace torchlet::ops::init