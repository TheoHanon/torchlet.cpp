#pragma once
#include <cstddef>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "dtype.h"
#include "index.h"
#include "rng.h"

struct Storage {
  void *data = nullptr;
  void (*deleter)(void *) = [](void *dt) { std::free(dt); };

  ~Storage() {
    if (data && deleter)
      deleter(data);
  };
};

class Tensor {

public:
  Tensor(const std::vector<std::size_t> &shape, const Dtype &dtype);
  Tensor() = default;

  static Tensor zeros(const std::initializer_list<std::size_t> &shape,
                      const Dtype &dtype);
  static Tensor zeros(const std::vector<std::size_t> &shape,
                      const Dtype &dtype);

  static Tensor ones(const std::initializer_list<std::size_t> &shape,
                     const Dtype &dtype);
  static Tensor ones(const std::vector<std::size_t> &shape, const Dtype &dtype);

  Tensor index(const std::initializer_list<std::size_t> &index) const;
  Tensor
  index(const std::initializer_list<torchlet::index::Slice> &index) const;

  Tensor permute(const std::size_t &idx1, const std::size_t &idx2) const;
  Tensor view(const std::vector<std::size_t> &new_shape) const;

  template <typename T>
  void assign_(const std::initializer_list<std::size_t> &index, T val);
  template <typename T> void fill_(T val);

  template <typename T> inline T *data_ptr() {
    return reinterpret_cast<T *>(m_storage->data);
  };
  template <typename T> inline const T *data_ptr() const {
    return reinterpret_cast<const T *>(m_storage->data);
  };
  inline std::shared_ptr<Storage> storage_ptr() const { return m_storage; };

  inline std::size_t elem_offset() const noexcept { return m_elem_offset; };
  inline const std::vector<std::size_t> &shape() const noexcept {
    return m_shape;
  };
  inline const std::vector<std::size_t> &strides() const noexcept {
    return m_strides;
  };
  inline Dtype dtype() const noexcept { return m_dtype; }
  inline std::size_t numel() const noexcept { return m_numel; }
  inline bool is_contiguous() const noexcept { return m_contiguous; };

private:
  Dtype m_dtype;
  std::vector<std::size_t> m_shape;
  std::vector<std::size_t> m_strides;
  std::size_t m_elem_offset;
  std::size_t m_numel;
  std::shared_ptr<Storage> m_storage;
  bool m_contiguous = true;

  Tensor(const std::vector<std::size_t> &shape,
         const std::vector<std::size_t> &strides,
         const std::size_t &elem_offset, const Dtype &dtype,
         const std::shared_ptr<Storage> &storage, const bool &contiguous);
};
