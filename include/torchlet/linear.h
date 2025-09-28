#pragma once
#include "kernel.h"
#include "tensor.h"

class Linear {
public:
  Linear(std::size_t in_features, std::size_t out_features, bool bias,
         const Dtype &dtype);
  Tensor forward(const Tensor &x);

  template <typename T>
  void normal_(T mean, T stdev, Generator &gen = Generator::global());
  template <typename T>
  void uniform_(T start, T end, Generator &gen = Generator::global());

  bool bias() { return m_has_bias; };

private:
  std::size_t in_features;
  std::size_t out_features;
  Tensor m_weights;
  Tensor m_bias;
  bool m_has_bias;
};
