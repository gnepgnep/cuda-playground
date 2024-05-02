#pragma once 

#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>


namespace base {

enum DataType {D_FLOAT32, D_INT32};

class GPUTensor{
public:
    
    GPUTensor(DataType dtype, const std::vector<int>& shape, bool allocateMemory);
    ~GPUTensor();

    void allocate_gpu_memory(size_t element_size);
    void deallocate_gpu_memory();
    void random_uniform_value();
    size_t calculate_size() const;

    void print_shape() const;
    std::vector<int> shape() const;

    template <typename T>
    T* data() const;
    template <typename T>
    void print_data() const;
    template <typename T>
    T* data_to_cpu() const;
    template <typename T>
    void data_to_gpu(T* cpu_data);

    GPUTensor relu() const;
    GPUTensor operator+(const GPUTensor& other) const;
    GPUTensor operator-(const GPUTensor& other) const;
    friend std::ostream& operator<<(std::ostream& os, const GPUTensor& tensor);

private:
    DataType dtype_;
    std::vector<int> shape_;
    bool allocated_;
    void* data_;

    template <typename T>
    void generate_random_uniform_value(size_t size);

};

void compare_GPUTensor(const GPUTensor& tensor1, const GPUTensor& tensor2);

std::ostream& operator<<(std::ostream& os, GPUTensor& tensor);


}
