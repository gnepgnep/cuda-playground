#include <iostream>
#include <vector>
#include <ctime>

enum DataType {D_FLOAT32, D_INT32};

class GPUTensor{
public:
    GPUTensor(DataType dtype, const std::vector<int>& shape, bool allocateMemory){
        dtype_ = dtype;
        shape_ = shape;
        allocated_ = allocateMemory;

        if (allocated_){
            allocate_gpu_memory();
        }
    }

    ~GPUTensor(){
        if (allocated_) {
            deallocate_gpu_memory();
        }
    }

    void allocate_gpu_memory(){
        std::cout << "Allocating GPU memory for tensor of shape: ";
        print_shape();
        std::cout << std::endl;
        
        size_t size = calculate_size();
        cudaError_t cudaStatus = cudaMalloc(&data_, size);
        if (cudaStatus != cudaSuccess){
            std::cerr << "CUDA memory allocation error " << cudaGetErrorString(cudaStatus) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void deallocate_gpu_memory(){
        cudaFree(data_);

        std::cout << "Dellocating GPU memory for tensor of shape: ";
        print_shape();
        std::cout << std::endl;
    }


    void random_uniform_value() {
        size_t size = calculate_size();
        switch (dtype_) {
            case D_FLOAT32:
                generate_random_uniform_value<float>(size);
                break;
            case D_INT32:
                generate_random_uniform_value<int>(size);
                break;
            default:
                std::cerr << "Unsupported data type" << std::endl;
                std::exit(EXIT_FAILURE);
        }
        std::cout << "Generated random uniform values for tensor of shape: ";
        print_shape();
        std::cout << std::endl;
    }

private:
    DataType dtype_;
    std::vector<int> shape_;
    bool allocated_;
    void* data_;

    void print_shape() const {
        std::cout << "(";
        for (size_t i = 0; i < shape_.size(); ++i){
            std::cout << shape_[i];
            if (i < shape_.size() - 1){
                std::cout << ",";
            }
        }
        std::cout << ")";
    }

    size_t calculate_size() const {
        size_t element_size = 0;
        switch (dtype_){
            case D_FLOAT32:
                element_size = sizeof(float);
                break;
            case D_INT32:
                element_size = sizeof(int);
                break;
            default:
                std::cerr << "Unsupported data type" << std::endl;
                std::exit(EXIT_FAILURE);
        }
    
        size_t size = element_size;
        for (int dim: shape_){
            size *= dim;
        }
        return size;
    }

    template <typename T>
    void generate_random_uniform_value(size_t size){
        T* host_data = new T[size];

        for (size_t i = 0; i < size; ++i) {
            host_data[i] = static_cast<T>(rand()) / RAND_MAX;
        }

        cudaMemcpy(data_, host_data, sizeof(T) * size, cudaMemcpyHostToDevice);
        
        delete[] host_data;
    }

};

int main(){
    int B = 1000;
    int N = 2000;

    std::srand(std::time(0));

    GPUTensor x(D_FLOAT32, {B,N}, true);
    GPUTensor bias(D_FLOAT32, {N}, true);
    GPUTensor y(D_FLOAT32, {B,N}, true);

    x.random_uniform_value();
    y.random_uniform_value();

}