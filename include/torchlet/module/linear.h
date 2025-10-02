#pragma once

#include <torchlet/core/rng.h>
#include <torchlet/core/tensor.h>

namespace torchlet::module {

class Linear {
public:
  Linear(std::size_t in_features, std::size_t out_features, bool bias,
         const torchlet::core::Dtype &dtype);

  Linear() = delete;

  torchlet::core::Tensor forward(const torchlet::core::Tensor &x) const;

  torchlet::core::Tensor &bias() {
    if (m_bias.storage_ptr() == nullptr) {
      throw std::runtime_error("The bias torchlet::core::Tensor is empty.");
    }
    return m_bias;
  };
  const bool &has_bias() const { return m_has_bias; };
  torchlet::core::Tensor &weights() { return m_weights; };

private:
  std::size_t in_features;
  std::size_t out_features;
  torchlet::core::Tensor m_weights;
  torchlet::core::Tensor m_bias;
  bool m_has_bias;
};

} // namespace torchlet::module
