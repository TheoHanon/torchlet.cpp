#pragma once
#include "rng.h"
#include "tensor.h"

namespace torchlet::init {

template <typename T>
void normal_(Tensor &tensor, T mean, T stdev,
             Generator &gen = Generator::global());
template <typename T>
void uniform_(Tensor &tensor, T start, T end,
              Generator &gen = Generator::global());

} // namespace torchlet::init