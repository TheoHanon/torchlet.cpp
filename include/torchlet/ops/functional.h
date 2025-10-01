#pragma once
#include <torchlet/core/tensor.h>

namespace torchlet::ops {

torchlet::core::Tensor linear(const torchlet::core::Tensor &x,
                              const torchlet::core::Tensor &weights,
                              const torchlet::core::Tensor &bias);

torchlet::core::Tensor gelu(const torchlet::core::Tensor &x);
torchlet::core::Tensor &log_softmax(const torchlet::core::Tensor &X);
torchlet::core::Tensor &softmax(const torchlet::core::Tensor &x);

} // namespace torchlet::ops