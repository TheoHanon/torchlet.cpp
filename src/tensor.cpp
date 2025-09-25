#include "tensor.h"


void validate_shape(std::vector<size_t> old_shape, std::vector<size_t> new_shape){

    size_t new_prod = 1; size_t old_prod = 1;

    for (size_t k = 0 ; k < new_shape.size(); k++) new_prod *= new_shape[k];
    for (size_t k = 0 ; k < old_shape.size(); k++) old_prod *= old_shape[k];

    if (old_prod != new_prod) throw std::invalid_argument("Shapes doesn't match.");
};

void validate_strides(const std::vector<size_t>& strides) {

    for (size_t i = 1; i < strides.size(); i++) {
        if (strides[i - 1] < strides[i]) throw std::runtime_error("Memory layout is not row major.");
    }
}


Tensor::Tensor(const std::vector<size_t> &shape, const Dtype &dtype) : m_dtype(dtype), m_shape(shape), m_byte_offset(0)
{

    m_strides = get_strides(shape);
    size_t n_bytes = nbytes(shape, dtype);

    m_storage = std::make_shared<Storage>();
    m_storage -> data = std::malloc(n_bytes);
    std::memset(m_storage -> data, 0, n_bytes);
};


Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<size_t>& strides, const size_t& byte_offset,const Dtype& dtype, const std::shared_ptr<Storage>& storage)
: m_dtype(dtype), m_shape(shape), m_strides(strides), m_byte_offset(byte_offset), m_storage(storage) {};



Tensor Tensor::index(const std::initializer_list<size_t>& index) {    

    if (index.size() != m_strides.size()) {
        throw std::runtime_error("Wrong indices size.");
    }

    size_t byte_offset = 0;
    size_t k = 0;

    for (auto& id : index) {
        if (id >= m_shape[k]) throw std::runtime_error("Index out of range.");
        byte_offset += m_strides[k++] * id;
    }

    byte_offset += m_byte_offset;

    std::vector<size_t> new_shape{1};
    std::vector<size_t> new_strides{1};

    return Tensor(new_shape, new_strides, byte_offset, m_dtype, m_storage);
    
};


Tensor Tensor::permute(const size_t& idx1, const size_t& idx2){

    if (idx1 >= m_shape.size() || idx2 >= m_shape.size()){
        throw std::runtime_error("Index out of range.");
    }


    std::vector<size_t> new_shape(m_shape);
    std::vector<size_t> new_strides(m_strides);
    
    std::swap(new_shape[idx1], new_shape[idx2]);
    std::swap(new_strides[idx1], new_strides[idx2]);
    
    return Tensor(new_shape, new_strides, m_byte_offset, m_dtype, m_storage);

};


Tensor Tensor::view(const std::vector<size_t>& new_shape) {
    validate_shape(m_shape, new_shape);
    validate_strides(m_strides);

    std::vector<size_t> new_strides = get_strides(new_shape);
    
    return Tensor(new_shape, new_strides, m_byte_offset ,m_dtype, m_storage);
};