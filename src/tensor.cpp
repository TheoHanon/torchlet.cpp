#include <torchcpp/torchcpp.h>

#include "detail/helpers.h"
#include "detail/validators.h"

Tensor::Tensor(const std::vector<size_t> &shape, const Dtype &dtype) : m_dtype(dtype), m_shape(shape), m_elem_offset(0)
{

    m_strides = torchcpp::detail::get_strides(shape);
    size_t n_bytes = torchcpp::detail::nbytes(shape, dtype);

    m_storage = std::make_shared<Storage>();
    m_storage->data = std::malloc(n_bytes);

};

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<size_t> &strides, const size_t &elem_offset, const Dtype &dtype, const std::shared_ptr<Storage> &storage)
    : m_dtype(dtype), m_shape(shape), m_strides(strides), m_elem_offset(elem_offset), m_storage(storage) {};

Tensor Tensor::index(const std::initializer_list<size_t> &index)
{

    if (index.size() != m_strides.size())
    {
        throw std::runtime_error("Wrong indices size.");
    }

    size_t new_elem_offset = torchcpp::detail::get_offset(index, m_shape, m_strides, m_elem_offset);
    
    std::vector<size_t> new_shape{1};
    std::vector<size_t> new_strides{1};

    return Tensor(new_shape, new_strides, new_elem_offset, m_dtype, m_storage);
};

template <typename T>
void Tensor::assign_(const std::initializer_list<size_t>& index, T val) {

    if (DTypeToCPPType<T>::dtype != m_dtype) {
        throw std::runtime_error("Type T does not match the type of dtype.");
    }

    size_t elem_offset = torchcpp::detail::get_offset(index, m_shape, m_strides, m_elem_offset);
    T* data_ptr = static_cast<T*> (m_storage -> data);
    data_ptr[elem_offset] = val;

};

template void Tensor::assign_(const std::initializer_list<size_t>& index, float);
template void Tensor::assign_(const std::initializer_list<size_t>& index, double);
template void Tensor::assign_(const std::initializer_list<size_t>& index, int32_t);
template void Tensor::assign_(const std::initializer_list<size_t>& index, int64_t);
template void Tensor::assign_(const std::initializer_list<size_t>& index, uint8_t);



//TODO
template <typename T>
void Tensor::normal_(T mean, T std){

    if (DTypeToCPPType<T>::dtype != m_dtype) {
        throw std::runtime_error("Type T does not match the type of dtype.");
    }

    return;
};

template void Tensor::normal_(float, float  );
template void Tensor::normal_(double, double);

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

    return Tensor(new_shape, new_strides, m_elem_offset, m_dtype, m_storage);
};

Tensor Tensor::view(const std::vector<size_t> &new_shape)
{

    torchcpp::detail::validate_shape(m_shape, new_shape);
    torchcpp::detail::validate_strides(m_strides);

    std::vector<size_t> new_strides = torchcpp::detail::get_strides(new_shape);

    return Tensor(new_shape, new_strides, m_elem_offset, m_dtype, m_storage);
};