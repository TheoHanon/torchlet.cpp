#pragma once
#include <cstdlib>
#include <stdexcept>

namespace torchlet::index {

struct Slice {
  std::size_t start;
  std::size_t end;

  Slice(std::size_t idx) : start(idx), end(idx + 1) {};
  Slice(std::size_t start, std::size_t end) : start(start), end(end) {
    if (end <= start) throw std::runtime_error("Slice end must be > start");
  };

  inline std::size_t range() const { return end - start; }
};

}  // namespace torchlet::index