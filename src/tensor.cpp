#include <torchlet/tensor.h>

#include "detail/helpers.h"
#include "detail/validators.h"

Tensor::Tensor(const std::vector<size_t> &shape, const Dtype &dtype) : m_dtype(dtype), m_shape(shape), m_elem_offset(0)
{

    m_strides = torchlet::detail::get_strides(shape);
    m_numel = torchlet::detail::numel(shape);
    size_t n_bytes = torchlet::detail::nbytes(shape, dtype);

    m_storage = std::make_shared<Storage>();
    m_storage->data = std::malloc(n_bytes);

};

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<size_t> &strides, const size_t &elem_offset, const Dtype &dtype, const std::shared_ptr<Storage> &storage, const bool& contiguous = true)
    : m_dtype(dtype), m_shape(shape), m_strides(strides), m_elem_offset(elem_offset), m_storage(storage), m_contiguous(contiguous) {
        m_numel = torchlet::detail::numel(shape);
    };


Tensor Tensor::zeros(const std::initializer_list<size_t>& shape, const Dtype& dtype) {

    Tensor t = Tensor(shape, dtype);
    void* data_ptr = t.data_ptr<void>();
    memset(data_ptr, 0, t.numel());

    return t;
}
Tensor Tensor::zeros(const std::vector<size_t>& shape, const Dtype& dtype) {

    Tensor t = Tensor(shape, dtype);
    void* data_ptr = t.data_ptr<void>();
    memset(data_ptr, 0, t.numel());

    return t;
}

Tensor Tensor::index(const std::initializer_list<size_t>& index)
{

    if (index.size() != m_strides.size())
    {
        throw std::runtime_error("Wrong indices size.");
    }

    size_t new_elem_offset = torchlet::detail::get_offset(index, m_shape, m_strides, m_elem_offset);
    
    std::vector<size_t> new_shape{1};
    std::vector<size_t> new_strides{1};

    return Tensor(new_shape, new_strides, new_elem_offset, m_dtype, m_storage);
};


Tensor Tensor::index(const std::initializer_list<torchlet::index::Slice>& index) {

    if (index.size() != m_strides.size())
    {
        throw std::runtime_error("Wrong indices size.");
    }

    std::vector<size_t> new_shape(m_shape.size());
    size_t new_elem_offset = m_elem_offset;
    size_t k = 0;
    for (const auto& idx : index){
        new_shape[k++] = idx.range();
        new_elem_offset += idx.start;
    }

    std::vector<size_t> new_strides(m_strides);
    return Tensor(new_shape, new_strides, new_elem_offset, m_dtype, m_storage, false);

};


template <typename T>
void Tensor::fill_(T val){

    if (!m_contiguous) throw std::runtime_error("The memory layout must be contiguous.");

    if (CPPTypeToDType<T>::dtype != m_dtype) {
        throw std::runtime_error("Type T does not match the type of dtype.");
    }

    size_t elem_offset = torchlet::detail::get_offset({static_cast<size_t>(0)}, m_shape, m_strides, m_elem_offset);
    size_t numel = torchlet::detail::numel(m_shape);

    T* data_ptr = reinterpret_cast<T*> (m_storage -> data);
    std::fill(data_ptr + elem_offset, data_ptr + elem_offset + numel, val);
};


template void Tensor::fill_(float val);
template void Tensor::fill_(double val);
template void Tensor::fill_(int32_t val);
template void Tensor::fill_(int64_t val);
template void Tensor::fill_(uint8_t val);



template <typename T>
void Tensor::assign_(const std::initializer_list<size_t>& index, T val) {

    if (CPPTypeToDType<T>::dtype != m_dtype) {
        throw std::runtime_error("Type T does not match the type of dtype.");
    }

    size_t elem_offset = torchlet::detail::get_offset(index, m_shape, m_strides, m_elem_offset);
    T* data_ptr = reinterpret_cast<T*> (m_storage -> data);
    data_ptr[elem_offset] = val;

};

template void Tensor::assign_(const std::initializer_list<size_t>& index, float);
template void Tensor::assign_(const std::initializer_list<size_t>& index, double);
template void Tensor::assign_(const std::initializer_list<size_t>& index, int32_t);
template void Tensor::assign_(const std::initializer_list<size_t>& index, int64_t);
template void Tensor::assign_(const std::initializer_list<size_t>& index, uint8_t);




Tensor Tensor::permute(const size_t &idx1, const size_t &idx2)
{

    if (idx1 >= m_shape.size() || idx2 >= m_shape.size())
    {
        throw std::runtime_error("Index out of range.");
    }

    std::vector<size_t> new_shape(m_shape);
    std::vector<size_t> new_strides(m_strides);

    std::swap(new_shape[idx1], new_shape[idx2]);
    std::swap(new_strides[idx1], new_strides[idx2]);

    return Tensor(new_shape, new_strides, m_elem_offset, m_dtype, m_storage, false);
};

Tensor Tensor::view(const std::vector<size_t> &new_shape)
{

    if (!m_contiguous) throw std::runtime_error("Memory is not contiguous.");
    torchlet::detail::validate_shape(m_shape, new_shape);

    std::vector<size_t> new_strides = torchlet::detail::get_strides(new_shape);

    return Tensor(new_shape, new_strides, m_elem_offset, m_dtype, m_storage);
};