#pragma once
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <random>

#include "dtype.h"
#include "rng.h"
#include "index.h"

using std::size_t;

struct Storage {
    void* data = nullptr;
    void (*deleter)(void*) = [](void* dt){std::free(dt);};

    ~Storage(){
        if (data && deleter) deleter(data);
    };
};

class Tensor {

    public:
        Tensor(const std::vector<size_t>& shape, const Dtype& dtype);
        Tensor() = default;

        static Tensor zeros(const std::initializer_list<size_t>& shape, const Dtype& dtype);
        static Tensor zeros(const std::vector<size_t>& shape, const Dtype& dtype);

        Tensor index(const std::initializer_list<size_t>& index);
        Tensor index(const std::initializer_list<torchlet::index::Slice>& index);

        Tensor permute(const size_t& idx1, const size_t& idx2);
        Tensor view(const std::vector<size_t>& new_shape);

        template <typename T> void assign_(const std::initializer_list<size_t>& index, T val);
        template <typename T> void fill_(T val);

        template <typename T> inline T* data_ptr() const {return reinterpret_cast<T*>(m_storage->data);};
        inline std::shared_ptr<Storage> storage_ptr() const {return m_storage;};
        
        inline size_t elem_offset() const noexcept {return m_elem_offset;};
        inline const std::vector<size_t>& shape() const noexcept {return m_shape;};
        inline const std::vector<size_t>& strides() const noexcept {return m_strides;};
        inline Dtype dtype()  const noexcept {return m_dtype;}
        inline size_t numel() const noexcept {return m_numel;}
        inline bool is_contiguous() const noexcept {return m_contiguous;};


    private:
        Dtype m_dtype; 
        std::vector<size_t> m_shape;
        std::vector<size_t> m_strides;
        size_t m_elem_offset;
        size_t m_numel;
        std::shared_ptr<Storage> m_storage;
        bool m_contiguous = true;


        Tensor(const std::vector<size_t>& shape, const std::vector<size_t>& strides, const size_t& elem_offset, const Dtype& dtype, const std::shared_ptr<Storage>& storage, const bool& contiguous);
};
