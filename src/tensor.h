#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <cstdlib>
#include <utility>

using std::size_t;

enum class Dtype {
    Float32,
    Float64, 
    Int32, 
    Int64, 
    UInt8,
};

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

        
        void* data_ptr() const {return storage_->data;};
        const std::shared_ptr<Storage>& storage_ptr() const {return storage_;};
        const std::vector<size_t>& shape() const {return this->shape_;};


    private:
        Dtype dtype_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t itemsize_;
        std::shared_ptr<Storage> storage_;

        Tensor(const std::vector<size_t>& shape, const std::vector<size_t>& strides, const Dtype& dtype, const std::shared_ptr<Storage>& storage);

        void get_itemsize();
        void get_strides();
};


#endif //TENSOR_H