#pragma once
#include "kernel.h"
#include "tensor.h"

class Linear {
public:
  Linear(std::size_t in_features, std::size_t out_features, bool bias,
         const Dtype &dtype);

  Linear() = delete;

  Tensor forward(const Tensor &x);

  template <typename T>
  void normal_(T mean, T stdev, Generator &gen = Generator::global());

  template <typename T>
  void uniform_(T start, T end, Generator &gen = Generator::global());

  Tensor &bias() {
    if (m_bias.storage_ptr() == nullptr) {
      throw std::runtime_error("The bias tensor is empty.");
    }
    return m_bias;
  };
  const bool &has_bias() { return m_has_bias; };
  Tensor &weights() { return m_weights; };

private:
  std::size_t in_features;
  std::size_t out_features;
  Tensor m_weights;
  Tensor m_bias;
  bool m_has_bias;
};
