#include <torchlet/init.h>



//TODO : Remove hard coded loops -> implement iterator.
namespace torchlet::init {

template <typename T>
void normal_(Tensor& tensor , T mean, T stdev, Generator& gen){

    if (CPPTypeToDType<T>::dtype != tensor.dtype()) {
        throw std::runtime_error("Type T does not match the type of dtype.");
    }

    std::normal_distribution<T> dist{mean, stdev};
    
    size_t elem_offset = tensor.elem_offset();
    T* data_ptr = tensor.data_ptr<T>() + elem_offset;


    if (tensor.is_contiguous()) {
        for (size_t idx = 0; idx< tensor.numel(); idx++) 
            data_ptr[idx] = dist(gen.engine());
    } else {

    std::vector<size_t> shape = tensor.shape();
    std::vector<size_t> strides = tensor.strides();
    
    for (size_t idx = 0; idx < tensor.numel(); idx++){
        size_t offset = elem_offset;
        size_t tmp = idx;
        for (size_t dim = shape.size(); dim-- > 0;) {
            size_t coord = tmp % shape[dim];
            tmp /= shape[dim];
            offset += coord * strides[dim]; 
        }
        data_ptr[offset] = dist(gen.engine());
    }}

    return;
};


template <typename T>
void uniform_(Tensor& tensor, T start, T end, Generator& gen) {

    if (CPPTypeToDType<T>::dtype != tensor.dtype()) {
        throw std::runtime_error("Type T does not match the type of dtype.");
    }

    std::uniform_real_distribution<T> dist{start, end};
    
    size_t elem_offset = tensor.elem_offset();
    T* data_ptr = tensor.data_ptr<T>() + elem_offset;


    if (tensor.is_contiguous()) {
        for (size_t idx = 0; idx< tensor.numel(); idx++) 
            data_ptr[idx] = dist(gen.engine());
    } else {

    std::vector<size_t> shape = tensor.shape();
    std::vector<size_t> strides = tensor.strides();
    
    for (size_t idx = 0; idx < tensor.numel(); idx++){
        size_t offset = elem_offset;
        size_t tmp = idx;
        for (size_t dim = shape.size(); dim-- > 0;) {
            size_t coord = tmp % shape[dim];
            tmp /= shape[dim];
            offset += coord * strides[dim]; 
        }
        data_ptr[offset] = dist(gen.engine());
    }}

};



template void normal_(Tensor& , float, float, Generator&);
template void normal_(Tensor& , double, double , Generator&);

template void uniform_(Tensor& , float, float, Generator&);
template void uniform_(Tensor& , double, double , Generator&);


}
