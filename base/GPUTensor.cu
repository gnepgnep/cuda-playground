#include "GPUTensor.h"

using namespace base;

GPUTensor::GPUTensor(DataType dtype, const std::vector<int>& shape, bool allocateMemory){
    dtype_ = dtype;
    shape_ = shape;
    allocated_ = allocateMemory;

    switch (dtype_){
        case D_FLOAT32:
            datacpu_ = malloc(calculate_size()*sizeof(float));
            break;
        case D_INT32:
            datacpu_ = malloc(calculate_size()*sizeof(int));
            break;
        default:
            std::cerr << "Unsupported data type" << std::endl;
            std::exit(EXIT_FAILURE);
    }

    if (allocated_){
        switch (dtype_){
            case D_FLOAT32:
                allocate_gpu_memory(sizeof(float));
                break;
            case D_INT32:
                allocate_gpu_memory(sizeof(int));
                break;
            default:
                std::cerr << "Unsupported data type" << std::endl;
                std::exit(EXIT_FAILURE);
        }
    }
}

GPUTensor::~GPUTensor(){
    if (allocated_) {
        deallocate_gpu_memory();
    }
}

void GPUTensor::allocate_gpu_memory(size_t element_size){
    std::cout << "Allocating GPU memory for tensor of shape: ";
    print_shape();
    std::cout << std::endl;
    
    CUDA_CHECK_ERROR(cudaMalloc(&data_, calculate_size()*element_size));
}

void GPUTensor::deallocate_gpu_memory(){
    CUDA_CHECK_ERROR(cudaFree(data_));

    std::cout << "Dellocating GPU memory for tensor of shape: ";
    print_shape();
    std::cout << std::endl;
}


void GPUTensor::random_uniform_value() {
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


void GPUTensor::print_shape() const {
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
T* GPUTensor::get_data(std::string mode) const {
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


template <typename T>
__global__ void print_kernel(T* gpuData, int numElements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numElements) {
        T element = gpuData[tid];
        printf("gpuData[%d] = %f\n", tid, static_cast<float>(element));
    }
}

template <typename T>
void GPUTensor::print_data() const {
    // T* typed_data = get_data<T>("cuda");

    T* cpu_data = get_data<T>("cpu");

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

    // delete[] cpu_data;

    // size_t size = calculate_size();
    // int blockSize = 256;
    // int gridSize = (size + blockSize - 1) / blockSize;
    // print_kernel<T><<<gridSize, blockSize>>>(typed_data, size);
    // cudaDeviceSynchronize();
}

template <typename T>
void GPUTensor::data_to_cpu(T* gpu_data) {
    CUDA_CHECK_ERROR(cudaMemcpy(datacpu_, gpu_data, calculate_size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void GPUTensor::data_to_gpu(T* cpu_data) {
    CUDA_CHECK_ERROR(cudaMemcpy(data_, cpu_data, calculate_size() * sizeof(T), cudaMemcpyHostToDevice));
}

std::vector<int> GPUTensor::shape() const {
    return shape_;
}

template <typename T>
T* GPUTensor::data() const {
    if (data_ == nullptr) {
        return nullptr;
    }

    return static_cast<T*>(data_);
}


size_t GPUTensor::calculate_size() const {

    size_t size = 1;
    for (int dim: shape_){
        size *= dim;
    }
    return size;
}

template <typename T>
void GPUTensor::generate_random_uniform_value(size_t size){
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

GPUTensor GPUTensor::relu() const {
    GPUTensor result(dtype_, shape_, true);
    float* input = get_data<float>("cpu");
    float* relu_res = result.get_data<float>("cpu");
    for (size_t i = 0; i < calculate_size(); ++i) {
        relu_res[i] = fmaxf(input[i], 0);
    }
    result.data_to_gpu(relu_res);

    return result;
}


GPUTensor GPUTensor::transpose() const {
    std::vector<int> reversed_shape(shape_.size());
    std::reverse_copy(shape_.begin(), shape_.end(), reversed_shape.begin());
    GPUTensor result(dtype_, reversed_shape, true);
    float* input = get_data<float>("cpu");
    float* relu_res = result.get_data<float>("cpu");
    int M = shape_[0] - 1;
    int N = shape_[1] - 1;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            relu_res[j * M + i] = input[i * N + j];
        }
    }
    result.data_to_gpu(relu_res);

    return result;
}

GPUTensor GPUTensor::operator+(const GPUTensor& other) const {
    GPUTensor result(dtype_, shape_, true);
    
    float* add1 = get_data<float>("cpu");
    float* add2 = other.get_data<float>("cpu");
    float* add_res = result.get_data<float>("cpu");
    size_t idx = calculate_size();
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            add_res[i * shape_[1] + j] = add1[i * shape_[1] + j] + add2[j];
        }
    }
    result.data_to_gpu(add_res);

    return result;
}

GPUTensor GPUTensor::operator-(const GPUTensor& other) const {
    GPUTensor result(dtype_, shape_, true);
    
    float* data1 = get_data<float>("cpu");
    float* data2 = other.get_data<float>("cpu");
    float* res = result.get_data<float>("cpu");
    size_t idx = calculate_size();
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            res[i * shape_[1] + j] = data1[i * shape_[1] + j] - data2[i * shape_[1] + j];
        }
    }
    result.data_to_gpu(res);

    return result;
}

void base::compare_GPUTensor(const GPUTensor& tensor1, const GPUTensor& tensor2) {
    GPUTensor diff = tensor1 - tensor2;
    float* diff_cpu = diff.get_data<float>("cpu");
    for (size_t i = 0; i < diff.calculate_size(); ++i) {
        if (diff_cpu[i] != 0) {
            printf("Find diff value in idx: %ld\n", i);
            delete[] diff_cpu;
            return;
        }
    }
    delete[] diff_cpu;
    printf("compare_GPUTensor result: same values\n");
}

std::ostream& base::operator<<(std::ostream& os, GPUTensor& tensor) {
    os << "Shape: ";
    tensor.print_shape();
    os << "\nData:\n";
    tensor.print_data<float>();

    return os;
}