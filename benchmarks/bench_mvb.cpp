#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <torchlet/torchlet.h>
#include <vector>

static inline bool approx_equal(float a, float b, float rel = 1e-4f,
                                float abs = 1e-7f) {
  float diff = std::fabs(a - b);
  if (diff <= abs)
    return true;
  return diff <= rel * std::max(std::fabs(a), std::fabs(b));
}

struct BenchResult {
  double ms;       // best time in milliseconds
  double gflops;   // derived GFLOP/s
  double gbytes_s; // rough effective GB/s
};

static inline double estimate_bytes(std::size_t m, std::size_t n) {
  const double bytes_W = 4.0 * double(m) * double(n);
  const double bytes_x = 4.0 * double(n);
  const double bytes_b = 4.0 * double(m);
  const double bytes_y = 4.0 * double(m);
  return bytes_W + bytes_x + bytes_b + bytes_y;
}

BenchResult
bench_kernel(void (*kernel)(const float *, const float *, const float *,
                            float *, std::size_t, std::size_t),
             const char *name, const std::vector<float> &W,
             const std::vector<float> &x, const std::vector<float> &b,
             std::vector<float> &y, std::size_t m, std::size_t n,
             int warmup_runs = 2, int trials = 5) {
  // Warm-ups
  for (int w = 0; w < warmup_runs; ++w) {
    kernel(W.data(), x.data(), b.empty() ? nullptr : b.data(), y.data(), m, n);
  }
  // Timed trials — take best
  double best_ms = 1e100;
  for (int t = 0; t < trials; ++t) {
    auto t0 = std::chrono::steady_clock::now();
    kernel(W.data(), x.data(), b.empty() ? nullptr : b.data(), y.data(), m, n);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best_ms = std::min(best_ms, ms);
  }

  const double flops = 2.0 * double(m) * double(n);
  const double seconds = best_ms / 1000.0;
  const double gflops = (flops / seconds) / 1e9;

  const double bytes = estimate_bytes(m, n);
  const double gbytes_s = (bytes / seconds) / 1e9;

  volatile float sink = 0.0f;
  for (std::size_t i = 0; i < m; ++i)
    sink += y[i];
  (void)sink;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << name << "  best: " << best_ms << " ms,  " << gflops
            << " GFLOP/s,  " << gbytes_s << " GB/s\n";

  return BenchResult{best_ms, gflops, gbytes_s};
}

int main(int argc, char **argv) {

  std::size_t m = 2048, n = 2048;
  if (argc >= 3) {
    m = static_cast<std::size_t>(std::stoull(argv[1]));
    n = static_cast<std::size_t>(std::stoull(argv[2]));
  }

  std::cout << "Matrix-Vector benchmark (m=" << m << ", n=" << n << ")\n";

  std::vector<float> W(m * n), x(n), b(m), y_scalar(m), y_blas(m);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto &v : W)
    v = dist(rng);
  for (auto &v : x)
    v = dist(rng);
  for (auto &v : b)
    v = dist(rng);

  mvb_kernel(W.data(), x.data(), b.data(), y_scalar.data(), m, n);
  mvb_blas_kernel(W.data(), x.data(), b.data(), y_blas.data(), m, n);

  for (std::size_t i = 0; i < m; ++i) {
    if (!approx_equal(y_scalar[i], y_blas[i])) {
      std::cerr << "Mismatch at row " << i << " rdiff="
                << std::fabs(y_scalar[i] - y_blas[i]) / std::fabs(y_scalar[i])
                << " scalar=" << y_scalar[i] << " blas=" << y_blas[i] << "\n";
    }
  }

  // Benchmarks
  auto r_scalar = bench_kernel(mvb_kernel, "Scalar", W, x, b, y_scalar, m, n);
  auto r_blas = bench_kernel(mvb_blas_kernel, "BLAS  ", W, x, b, y_blas, m, n);

  double speedup_blas = r_scalar.ms / r_blas.ms;

  std::cout << std::setprecision(2)
            << "Speedup (BLAS / Scalar): " << speedup_blas << "×\n";

  return 0;
}