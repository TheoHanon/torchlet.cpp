#pragma once
#include <random>

class Generator {

public:
  static Generator &global() noexcept {
    static Generator g;
    return g;
  };

  void manual_seed(std::uint32_t s) {
    m_seed = s;
    m_engine.seed(s);
  };

  Generator() = default;
  explicit Generator(std::uint32_t s) : m_seed(s), m_engine(s) {};

  std::mt19937 &engine() noexcept { return m_engine; } // for drawing
  const std::mt19937 &engine() const noexcept { return m_engine; } // read-only

private:
  std::uint32_t m_seed{static_cast<std::uint32_t>(std::random_device{}())};
  std::mt19937 m_engine{m_seed};

  Generator(const Generator &) = delete;
  Generator(Generator &&) = delete;
  Generator &operator=(const Generator &) = delete;
  Generator &operator=(Generator &&) = delete;
};
