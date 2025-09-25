#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <cstdlib>
#include <utility>

#include "dtype.h"

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

        Tensor index(const std::initializer_list<size_t>& index);
        Tensor permute(const size_t& idx1, const size_t& idx2);
        Tensor view(const std::vector<size_t>& new_shape);

        void* data_ptr() const {return m_storage->data;};
        std::shared_ptr<Storage> storage_ptr() const {return m_storage;};
        const std::vector<size_t>& shape() const {return m_shape;};
        size_t byte_offset() const {return m_byte_offset;};


    private:
        Dtype m_dtype; 
        std::vector<size_t> m_shape;
        std::vector<size_t> m_strides;
        size_t m_byte_offset;
        std::shared_ptr<Storage> m_storage;

        Tensor(const std::vector<size_t>& shape, const std::vector<size_t>& strides, const size_t& byte_offset, const Dtype& dtype, const std::shared_ptr<Storage>& storage);
};


#endif //TENSOR_H