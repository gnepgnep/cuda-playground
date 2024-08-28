#pragma once 
#include "GPUTensor.cuh"

// #define Debug 1

// using namespace base;

template <typename T>
GPUTensor<T>::GPUTensor(const std::vector<int>& shape, bool allocateMemory){
    shape_ = shape;
    allocated_ = allocateMemory;

    datacpu_ = malloc(calculate_size()*sizeof(T));

    if (allocated_){
        allocate_gpu_memory(sizeof(T));
    }
}

template <typename T>
GPUTensor<T>::~GPUTensor(){
    if (allocated_) {
        deallocate_gpu_memory();
    }
    if (datacpu_ != nullptr) {
        free(datacpu_);
        datacpu_ = nullptr; // Optional: Set pointer to null after freeing
    }
}

template <typename T>
void GPUTensor<T>::allocate_gpu_memory(size_t element_size){
#ifdef Debug
    std::cout << "Allocating GPU memory for tensor of shape: ";
    print_shape();
    std::cout << std::endl;
#endif
    CUDA_CHECK_ERROR(cudaMalloc(&data_, calculate_size()*element_size));
}

template <typename T>
void GPUTensor<T>::deallocate_gpu_memory(){
    CUDA_CHECK_ERROR(cudaFree(data_));
#ifdef Debug
    std::cout << "Dellocating GPU memory for tensor of shape: ";
    print_shape();
    std::cout << std::endl;
#endif
}

template <typename T>
void GPUTensor<T>::random_uniform_value() {
    size_t size = calculate_size();
    generate_random_uniform_value(size);
#ifdef Debug
    std::cout << "Generated random uniform values for tensor of shape: ";
    print_shape();
    std::cout << std::endl;
#endif
}

template <typename T>
void GPUTensor<T>::print_shape() const {
    std::cout << "(";
    for (size_t i = 0; i < shape_.size(); ++i){
        std::cout << shape_[i];
        if (i < shape_.size() - 1){
            std::cout << ",";
        }
    }
    std::cout << ")";
}

template <typename T>
T* GPUTensor<T>::get_data(std::string mode) const {
    if (mode == "cpu") {
        if (datacpu_ == nullptr) {
            return nullptr;
        }
        return static_cast<T*>(datacpu_);
    } else if (mode == "cuda") {
        if (data_ == nullptr) {
            return nullptr;
        }
        return static_cast<T*>(data_);
    } else {
        std::cerr << "Unsupported device type" << std::endl;
        std::exit(EXIT_FAILURE);

    }
}

// template int* GPUTensor<T>::get_data<int>(std::string mode) const;
// template float* GPUTensor<T>::get_data<float>(std::string mode) const;


template <typename T>
void GPUTensor<T>::data_to_cpu() {
    CUDA_CHECK_ERROR(cudaMemcpy(datacpu_, data_, calculate_size() * sizeof(T), cudaMemcpyDeviceToHost));
}
// template void GPUTensor<T>::data_to_cpu<int>();
// template void GPUTensor<T>::data_to_cpu<float>();

template <typename T>
void GPUTensor<T>::data_to_gpu() {
    CUDA_CHECK_ERROR(cudaMemcpy(data_, datacpu_, calculate_size() * sizeof(T), cudaMemcpyHostToDevice));
}
// template void GPUTensor<T>::data_to_gpu<int>();
// template void GPUTensor<T>::data_to_gpu<float>();

template <typename T>
std::vector<int> GPUTensor<T>::shape() const {
    return shape_;
}

template <typename T>
size_t GPUTensor<T>::calculate_size() const {

    size_t size = 1;
    for (int dim: shape_){
        size *= dim;
    }
    return size;
}

template <typename T>
void GPUTensor<T>::generate_random_uniform_value(size_t size){
    T* host_data = new T[size];

    for (size_t i = 0; i < size; ++i) {
        T raw_rand = static_cast<T>(rand()) / RAND_MAX;
        host_data[i] = (raw_rand - 0.5) * 2;
    }
    memcpy(datacpu_, host_data, sizeof(T) * size);

    if (allocated_) {
        CUDA_CHECK_ERROR(cudaMemcpy(data_, host_data, sizeof(T) * size, cudaMemcpyHostToDevice));
    }
    
    delete[] host_data;
}



template <typename T>
__global__ void print_kernel(T* gpuData, int numElements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numElements) {
        T element = gpuData[tid];
        printf("gpuData[%d] = %f", tid, static_cast<float>(element));
    }
}

template <typename T>
void GPUTensor<T>::print_data() {
    // T* typed_data = get_data<T>("cuda");
    data_to_cpu();
    T* cpu_data = get_data("cpu"); 

    if (shape_.size() == 2) {
        for (int i = 0; i < shape_[0]; ++i) {
            std::cout << "[";
            for (int j = 0; j < shape_[1]; ++j) {
                std::cout << cpu_data[i * shape_[1] + j];
                if (j < shape_[1] - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]";
            if (i < shape_[0] - 1) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    } else {
        std::cout << "[";
        for (int i = 0; i < shape_[0]; ++i) {
            std::cout << cpu_data[i];
            if (i < shape_[0] - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        std::cout << std::endl;
    }


    // size_t size = calculate_size();
    // int blockSize = 1;
    // int gridSize = size;
    // print_kernel<T><<<gridSize, blockSize>>>(get_data("cuda"), size);
    // cudaDeviceSynchronize();
}

template <typename T>
GPUTensor<T> GPUTensor<T>::relu() const {
    GPUTensor result(shape_, true);
    float* input = get_data("cpu");
    float* relu_res = result.get_data("cpu");
    for (size_t i = 0; i < calculate_size(); ++i) {
        relu_res[i] = fmaxf(input[i], 0);
    }
    result.data_to_gpu();

    return result;
}

template <typename T>
GPUTensor<T> GPUTensor<T>::transpose() const {
    std::vector<int> reversed_shape(shape_.size());
    std::reverse_copy(shape_.begin(), shape_.end(), reversed_shape.begin());
    GPUTensor result(reversed_shape, true);
    T* input = get_data("cpu");
    T* result_cpu = result.get_data("cpu");
    int M = shape_[0];
    int N = shape_[1];  
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            result_cpu[j * M + i] = input[i * N + j];
        }
    }
    result.data_to_gpu();

    return result;
}

template <typename T>
GPUTensor<T> GPUTensor<T>::operator+(const GPUTensor& other) const {
    GPUTensor result(shape_, true);
    
    float* add1 = get_data("cpu");
    float* add2 = other.get_data("cpu");
    float* add_res = result.get_data("cpu");
    size_t idx = calculate_size();
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            add_res[i * shape_[1] + j] = add1[i * shape_[1] + j] + add2[j];
        }
    }
    result.data_to_gpu();

    return result;
}

template <typename T>
GPUTensor<T> GPUTensor<T>::operator-(const GPUTensor& other) const {
    GPUTensor result(shape_, true);
    
    float* data1 = get_data("cpu");
    float* data2 = other.get_data("cpu");
    float* res = result.get_data("cpu");
    size_t idx = calculate_size();
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            res[i * shape_[1] + j] = data1[i * shape_[1] + j] - data2[i * shape_[1] + j];
        }
    }
    result.data_to_gpu();

    return result;
}

template <typename T>
void compare_GPUTensor(const GPUTensor<T>& tensor1, const GPUTensor<T>& tensor2) {
    GPUTensor diff = tensor1 - tensor2;
    float* diff_cpu = diff.get_data("cpu");
    for (size_t i = 0; i < diff.calculate_size(); ++i) {
        if (diff_cpu[i] != 0) {
            printf("Find diff value in idx: %ld\n", i);
            return;
        }
    }
    printf("compare_GPUTensor result: same values\n");
}

template <typename T>
std::ostream& operator<<(std::ostream& os, GPUTensor<T>& tensor) {
    os << "Shape: ";
    tensor.print_shape();
    os << "\nData:\n";
    tensor.print_data();

    return os;
}