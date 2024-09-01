#pragma once
#include <iostream>

extern __shared__ float dynamic_shared_data[];
template <typename T>
__global__ void ReduceKernel1(const T *x, T *y, const int M, const int N) {
    int tid = threadIdx.x;
    int mi = blockIdx.x;
    float *data = dynamic_shared_data;
    data[tid] = tid < N ? (float)x[mi * N + tid]: 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s*=2) {
        if (tid + s < N) data[tid] += data[tid + s];
        __syncthreads();
    }
    if(tid == 0) y[tid] = data[0];
}

template <typename T>
void ReduceRun1(const T *x, T *y, const int M, const int N, cudaStream_t stream) {
    int block_size = CalcBlockSize(N);
    int shared_mem_size = sizeof(float) * block_size;
    ReduceKernel1<float><<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}