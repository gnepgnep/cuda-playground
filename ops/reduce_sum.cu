/*
reduce_kernel1: origin implemetation
reduce_kernel2: reindex tid increase working thread in each wrap
reduce_kernel3: reduce bank conflict
reduce_kernel4: reduce shared memory, eliminate thread that only read data
reduce_kernel5: unravel the loop for the last warp, forceinline avoid shared memory cache
reduce_kernel6: use warp sync function
reduce_kernel7: reduce dim > 1024
*/

#include "kernels.cuh"

inline int CalcBlockSize(const int N) {
    int block_size = 32;
    while (block_size < N && block_size < 1024) {
        block_size <<= 1;
    }
    return block_size;
}

extern __shared__ float  dynamic_shared_buff[];

template <typename T>
__global__ void ReduceKernel1(const T *x, T *y, const int M, const int N) {
    int tid = threadIdx.x;
    int mi = blockIdx.x;
    float *data = dynamic_shared_buff;
    data[tid] = tid < N ? (float)x[mi * N + tid]: 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s*=2) {
        if (tid % (2*s) == 0) data[tid] += data[tid + s];
        __syncthreads();
    }
    if(tid == 0) y[mi] = data[0];
}

void ReduceFun1(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    int block_size = CalcBlockSize(N);
    int shared_mem_size = sizeof(float) * block_size;
    ReduceKernel1<float><<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}


template <typename T>
__global__ void ReduceKernel2(const T* x, T* y, const int M, const int N) {
    int tid = threadIdx.x;
    int mi = blockIdx.x;
    float *data = dynamic_shared_buff;
    data[tid] = tid < N ? (float)x[mi * N + tid] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s*=2) {
        int index = tid * s * 2;
        if (index < blockDim.x) data[index] += data[index + s];
        __syncthreads();
    }
    if (tid == 0) y[mi] = data[0];
}

void ReduceFun2(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    int block_size = CalcBlockSize(N);
    int shared_mem_size = sizeof(float) * block_size;
    ReduceKernel2<<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}


template <typename T>
__global__ void ReduceKernel3(const T* x, T* y, const int M, const int N) {
    int tid = threadIdx.x;
    int mi = blockIdx.x;
    float *data = dynamic_shared_buff;
    data[tid] = tid < N ? (float)x[mi * N + tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) data[tid] += data[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[mi] = data[0];
}

void ReduceFun3(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    int block_size = CalcBlockSize(N);
    int shared_mem_size = sizeof(float) * block_size;
    ReduceKernel3<<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}


template <typename T>
__global__ void ReduceKernel4(const T* x, T* y, const int M, const int N) {
    int tid = threadIdx.x;
    int mi = blockIdx.x;
    float *data = dynamic_shared_buff;
    data[tid] = (tid < N ? (float)x[mi * N + tid] : 0.0f) + \
                (tid + blockDim.x < N ? (float)x[mi * N + tid + blockDim.x] : 0.0f);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) data[tid] += data[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[mi] = data[0];
}

void ReduceFun4(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    int block_size = CalcBlockSize(N);
    int shared_mem_size = sizeof(float) * block_size;
    ReduceKernel4<<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}