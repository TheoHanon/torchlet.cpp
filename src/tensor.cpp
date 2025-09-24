#include "tensor.h"


void validate_shape(std::vector<size_t> old_shape, std::vector<size_t> new_shape){

    if (new_shape.size() != old_shape.size()) throw std::invalid_argument("Invalid shape.");
    
    size_t new_prod = 1; size_t old_prod = 1;

    for (size_t k = 0 ; k < new_shape.size(); k++) {new_prod *= new_shape[k]; old_prod *= old_shape[k];}

    if (old_prod != new_prod) throw std::invalid_argument("Shapes doesn't match.");
};



void Tensor::get_itemsize() {
    switch (dtype_) {
        case Dtype::Float32: itemsize_ = 4;break;
        case Dtype::Float64: itemsize_ = 8;break;
        case Dtype::Int32:   itemsize_ = 4;break;
        case Dtype::Int64:   itemsize_ = 8;break;
        case Dtype::UInt8:   itemsize_ = 1;break;
        default : throw std::runtime_error("Invalid type.");
    }
}

void Tensor::get_strides() {

    strides_.reserve(shape_.size());
    strides_.resize(shape_.size());
    std::uint64_t stride = 1;

    for (int i = shape_.size() - 1; i >= 0; i--)
    {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, const Dtype &dtype) : dtype_(dtype), shape_(shape)
{
    get_itemsize();
    get_strides();

    size_t size = static_cast<size_t> (itemsize_ * strides_[0] * shape_[0]);
    storage_ = std::make_shared<Storage>();
    storage_ -> data = std::malloc(size);
};


Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<size_t>& strides, const Dtype& dtype, const std::shared_ptr<Storage>& storage)
: dtype_(dtype), shape_(shape), strides_(strides), storage_(storage) {
    get_itemsize();
};



Tensor Tensor::index(const std::initializer_list<size_t>& index) {    

    if (index.size() != strides_.size()) {
        throw std::runtime_error("Wrong indices size.");
    }

    size_t idx = 0;
    size_t k = 0;

    for (auto id : index) {
        if (id >= shape_[k]) throw std::runtime_error("Index out of range.");
        idx += strides_[k++] * id;
    }

    std::vector<size_t>&& new_shape{1};
    std::vector<size_t>&& new_strides{idx * itemsize_};

    return Tensor(new_shape, new_strides, dtype_, storage_);
    
};


Tensor Tensor::permute(const size_t& idx1, const size_t& idx2){

    if (idx1 >= shape_.size() || idx2 >= shape_.size()){
        throw std::runtime_error("Index out of range.");
    }


    std::vector<size_t> new_shape(shape_);
    std::vector<size_t> new_strides(strides_);
    
    std::swap(new_shape[idx1], new_shape[idx2]);
    std::swap(new_strides[idx1], new_strides[idx2]);
    
    return Tensor(new_shape, new_strides, dtype_, storage_);

};


//TODO
Tensor Tensor::view(const std::vector<size_t>& new_shape) {

    validate_shape(shape_, new_shape);
    std::vector<size_t> permutations(new_shape.size());




};