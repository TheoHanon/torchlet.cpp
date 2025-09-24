#include <iostream>
#include <cstddef> 
#include <bitset>
#include <vector>
#include "src/tensor.h"


using namespace std;

int main(){
    
    vector<size_t> shape{2, 3, 5};
    Dtype dtype = Dtype::Int32;

    Tensor t1 = Tensor(shape, dtype);
    Tensor t2 = t1.permute(2, 1);


    std::cout << "{";
    for (auto s : t1.shape()){
        std::cout << s << " ";
    }

    std::cout << "}" << std::endl;

    std::cout << "{";
    for (auto s : t2.shape()){
        std::cout << s << " ";
    }

    std::cout << "}" << std::endl;
    
    return 0;

}