#include "kernels.cuh"

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

int AddBiasRelu0(const float* x, const float* bias, float* y, 
                        const int B, const int N, cudaStream_t stream) {
    dim3 grid(B);
    dim3 block(std::min(N, 1024));
    AddBiasRelu0Kernel<float><<<grid, block, 0, stream>>>(x, bias, y, N);
    if (cudaGetLastError() != cudaSuccess){
        printf("lauch kernel failed\n");
        return -1;
    }
    return 0;
}

int AddBiasRelu1(const float* x, const float* bias, float* y, 
                        const int B, const int N, cudaStream_t stream) {
    int Num = B * N;
    dim3 block(std::min(Num, 1024));
    dim3 grid((Num + block.x - 1) / block.x);
    AddBiasReluKernel1<float><<<grid, block, 0, stream>>>(x, bias, y, N, Num);
    if (cudaGetLastError() != cudaSuccess){
        printf("lauch kernel failed\n");
        return -1;
    }
    return 0;
}