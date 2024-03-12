#include "use_tensor.h"

template <typename T>
__global__ void AddBiasRelu0Kernel(const T* x, const T* bias, T* y, const int N) {
    auto bid = blockIdx.x;
    auto tid = threadIdx.x;
    for (auto id = tid; id < N; id += blockDim.x) {
        float v = (float)x[bid * N + id] + (float)bias[id];
        v = fmaxf(v, 0);
        y[bid * N + id] = (T)v;
    }
}

int AddBiasRelu0(const float* x, const float* bias, float* y, 
                        const int B, const int N, cudaStream_t stream) {
    dim3 grid(B);
    dim3 block(std::min(N, 1024));
    AddBiasRelu0Kernel<float><<<grid, block, 0, stream>>>(x, bias, y, N);
    if (cudaGetLastError() != cudaSuccess){
        printf("lauch kernel failed");
        return -1;
    }
    return 0;
}

template <typename T>
__global__ void AddBiasReluKernel1(const T* x, const T* bias, T* y, const int N, const int Num) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < Num) {
        int bias_id = id % N;
        float v = (float)x[id] + (float)bias[bias_id];
        v = fmaxf(v, 0);
        y[id] = (T)v;
    }
}

int AddBiasRelu1(const float* x, const float* bias, float* y, 
                        const int B, const int N, cudaStream_t stream) {
    int Num = B * N;
    dim3 block(std::min(Num, 1024));
    dim3 grid((Num + block.x - 1) / block.x);
    AddBiasReluKernel1<float><<<grid, block, 0, stream>>>(x, bias, y, N, Num);
    if (cudaGetLastError() != cudaSuccess){
        printf("lauch kernel failed");
        return -1;
    }
    return 0;
}

int TestAddBiasRelu(const int B, const int N, const int method_id, const int run_times = 100) {
    GPUTensor x(D_FLOAT32, {B,N}, true);
    GPUTensor bias(D_FLOAT32, {N}, true);
    GPUTensor y(D_FLOAT32, {B,N}, true);
    x.random_uniform_value();
    bias.random_uniform_value();
    y.random_uniform_value();
    cudaDeviceSynchronize();

    auto expect_y = (x + bias).relu();
    if (B * N < 256){
        std::cout << "x: " << x;
        std::cout << "bias: " << bias;
        std::cout << "expect y: " << expect_y;
    }

    std::cout << std::endl;
    std::cout << "===== ";
    std::cout << "START method id: " << method_id;
    std::cout <<  "===== " << std::endl;
    if (method_id == 0) {
        AddBiasRelu0(x.data<float>(), bias.data<float>(), y.data<float>(), B, N, nullptr);
        cudaDeviceSynchronize();
        std::cout << "y: " << y;
        compare_GPUTensor(expect_y, y);
    } else if (method_id == 1) {
        AddBiasRelu1(x.data<float>(), bias.data<float>(), y.data<float>(), B, N, nullptr);
        cudaDeviceSynchronize();
        std::cout << "y: " << y;
        compare_GPUTensor(expect_y, y);        
    }
    std::cout << "===== ";
    std::cout << "END method id: " << method_id;
    std::cout <<  "===== " << std::endl;
    std::cout << std::endl;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaEventBlockingSync);
    std::vector<GPUTensor *> io_list{&x, &bias, &y};
    auto call_fun0 = [&](const std::vector<GPUTensor *> &io_list){
        AddBiasRelu0(io_list[0]->data<float>(), io_list[1]->data<float>(),
                     io_list[2]->data<float>(), B, N, stream);
    };
    auto call_fun1 = [&](const std::vector<GPUTensor *> &io_list){
        AddBiasRelu1(io_list[0]->data<float>(), io_list[1]->data<float>(),
                     io_list[2]->data<float>(), B, N, stream);
    };
    float mem_size_GB = (float)sizeof(float) * (B * N * 2 + N) / (1024 * 1024 * 1024);
    switch (method_id) {
        case 0:
            RunStreamTrueBandwidth(call_fun0, io_list, run_times, stream, mem_size_GB);
            break;
        case 1:
            RunStreamTrueBandwidth(call_fun1, io_list, run_times, stream, mem_size_GB);
            break;
    }
    return 0;
}


int main(int argc, char** argv){
    std::srand(std::time(0));

    int B = 2;
    int N = 5;
    int run_times = 100;
    int device_id = 6;
    int method_id = 0;

    for (int i = 1; i < argc; ++i) {
        std::string opt(argv[i]);
        if (opt == "-B") 
            B = std::stoi(argv[++i]);
        else if (opt == "-N") 
            N = std::stoi(argv[++i]);
        else if (opt == "-R") 
            run_times = std::stoi(argv[++i]);
        else if (opt == "-M") 
            method_id = std::stoi(argv[++i]);
    }
    cudaSetDevice(device_id);
    TestAddBiasRelu(B, N, method_id, run_times);


    return 0;
}