#include <iostream>
#include <cstddef> 
#include <bitset>
#include <vector>
#include "src/tensor.h"


using namespace std;

template <typename T>
void rprint(Tensor& tensor, size_t dim, size_t& index){

    std::vector<size_t> shape = tensor.shape();
    void* data_ptr = tensor.data_ptr();
  
    if (dim == shape.size() -1) {
        std::cout << "[ ";
        for (size_t idx = 0; idx < shape[dim]; idx ++) {
            std::cout << *(reinterpret_cast<T*>(data_ptr) + index);
            if (idx != shape[dim] - 1) std::cout << ", ";
            index++;
        }
        std::cout << " ]";
    } else {
        std::cout << "[";
        for (size_t k = 0; k < shape[dim]; k++) {
            rprint<T>(tensor, dim + 1, index);
            if (k != shape[dim] - 1) std::cout << ", ";
        }   
        std::cout<< "]";
    }
};


template <typename T> 
void print(Tensor& tensor) {

    size_t index = tensor.byte_offset();
    size_t dim = 0;
    rprint<T>(tensor, dim, index);

};


template <typename T>
void print_raw(Tensor& tensor) {
    
    size_t len = 1;
    size_t offset = tensor.byte_offset();
    std::vector<size_t> shape = tensor.shape();
    T* data_ptr = reinterpret_cast<T*>(tensor.data_ptr());

    for (const auto& sh : shape) len *= sh;
    std::cout << "[";
    for (size_t k = 0; k < len; k++) {
        std::cout << *(data_ptr + k + offset) << " ";
    }

    std::cout << "]" << std::endl;
}

int main(){
    
    vector<size_t> shape{2, 3, 5};
    Dtype dtype = Dtype::Int32;

    Tensor t1 = Tensor(shape, dtype);
    Tensor ttemp = t1.index({0, 1, 3});

    std::cout << "Byte offeset: " << t1.byte_offset() << std::endl;
    std::cout<< "Count : " << t1.storage_ptr().use_count() << std::endl;
    
    // int32_t* data_ptr = reinterpret_cast<int32_t*>(t1.data_ptr());

    // for ()

    std::cout << ttemp.byte_offset() << std::endl;
    
    int32_t* ptr = reinterpret_cast<int32_t*>(ttemp.data_ptr()) + ttemp.byte_offset();
    
    *ptr = 10;
    print<int32_t>(t1);
    print<int32_t>(ttemp);
    std::cout << "\n\n";
    
    // Tensor t2 = t1.view({6, 5});
    // print<int32_t>(t2);
    

    std::cout << "{";
    for (auto s : t1.shape()){
        std::cout << s << " ";
    }

    std::cout << "}" << std::endl;

    // std::cout << "{";
    // for (auto s : t2.shape()){
    //     std::cout << s << " ";
    // }

    std::cout << "}" << std::endl;
    
    return 0;

}