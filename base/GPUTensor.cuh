#pragma once 

#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPU Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}


template <typename T>
class GPUTensor{
public:
    
    GPUTensor(const std::vector<int>& shape, bool allocateMemory);
    ~GPUTensor();

    void allocate_gpu_memory(size_t element_size);
    void deallocate_gpu_memory();
    void random_uniform_value();
    size_t calculate_size() const;
    void print_shape() const;
    std::vector<int> shape() const;


    T* get_data(std::string mode) const;
    void print_data();
    void data_to_cpu();
    void data_to_gpu();

    GPUTensor relu() const;
    GPUTensor transpose() const;
    GPUTensor reduce_sum() const;
    GPUTensor operator+(const GPUTensor& other) const;
    GPUTensor operator-(const GPUTensor& other) const;
    friend std::ostream& operator<<(std::ostream& os, const GPUTensor<int>& tensor);
    friend std::ostream& operator<<(std::ostream& os, const GPUTensor<float>& tensor);

private:
    std::vector<int> shape_;
    bool allocated_;
    void* data_;
    void* datacpu_;

    void generate_random_uniform_value(size_t size);

};

template <typename T>
void compare_GPUTensor(const GPUTensor<T>& tensor1, const GPUTensor<T>& tensor2);

template <typename T>
std::ostream& operator<<(std::ostream& os, GPUTensor<T>& tensor);

#include "GPUTensor.hpp" 



