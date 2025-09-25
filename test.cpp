#include <iostream>
#include <cstddef> 
#include <bitset>
#include <vector>
#include <random>

#include <torchcpp/torchcpp.h>


using namespace std;

template <typename T>
void rprint(Tensor& tensor, size_t dim, size_t& index){

    std::vector<size_t> shape = tensor.shape();
    T* data_ptr = tensor.data_ptr<T>();
  
    if (dim == shape.size() -1) {
        std::cout << "[ ";
        for (size_t idx = 0; idx < shape[dim]; idx ++) {
            std::cout << *(data_ptr + index);
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
    std::cout << std::endl;
};


template <typename T>
void print_raw(Tensor& tensor) {
    
    size_t len = 1;
    size_t offset = tensor.byte_offset();
    std::vector<size_t> shape = tensor.shape();
    T* data_ptr = tensor.data_ptr<T>();

    for (const auto& sh : shape) len *= sh;
    std::cout << "[";
    for (size_t k = 0; k < len; k++) {
        std::cout << *(data_ptr + k + offset) << " ";
    }

    std::cout << "]" << std::endl;
}

int main(){

    Generator gen = Generator();
    std::normal_distribution<double> dist{0.0, 1.0};


    std::cout << "Random number :" << dist(gen.engine()) << std::endl; 
    std::cout << "Random number :" << dist(gen.engine()) << std::endl; 
    

    vector<size_t> shape{2, 3, 5};
    Dtype dtype = Dtype::Float32;

    Tensor t1 = Tensor(shape, dtype);
    Tensor ttemp = t1.index({0, 1, 3});
    Tensor t2 = t1.view({30});


    t1.assign_({0, 1, 3}, 10.0f);
    t1.assign_({0, 1, 2}, 10.0f);

    print<float>(t1);
    print<float>(t2);
    print<float>(ttemp);
    std::cout << "\n\n";
    
    
    
    return 0;

}