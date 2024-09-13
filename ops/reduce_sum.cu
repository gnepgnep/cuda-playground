/*
reduce_kernel1: origin implemetation
reduce_kernel2: reindex tid increase working thread in each wrap
reduce_kernel3: reduce bank conflict
reduce_kernel4: reduce shared memory, eliminate thread that only read data
reduce_kernel5: unravel the loop for the last warp avoid syncthread, forceinline avoid shared memory cache
reduce_kernel6: use warp sync 
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
    ReduceKernel2<float><<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
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
    ReduceKernel3<float><<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
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
    int block_size = CalcBlockSize(N) / 2;
    int shared_mem_size = sizeof(float) * block_size;
    ReduceKernel4<float><<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}

// use volatile to avoid cache shared memory in register, ensure data update consistent
__device__ __forceinline__ void WarpReduce(volatile float* data, int tid) {
    data[tid] += data[tid + 32];
    data[tid] += data[tid + 16];
    data[tid] += data[tid + 8];
    data[tid] += data[tid + 4];
    data[tid] += data[tid + 2];
    data[tid] += data[tid + 1];
}

template <typename T>
__global__ void ReduceKernel5(const T* x, T* y, const int M, const int N) {
    int tid = threadIdx.x;
    int mi = blockIdx.x;
    float *data = dynamic_shared_buff;
    data[tid] = (tid < N ? (float)x[mi * N + tid] : 0.0f) + \
                (tid + blockDim.x < N ? (float)x[mi * N + tid + blockDim.x] : 0.0f);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>=1) {
        if (tid < s) data[tid] += data[tid + s];
        __syncthreads();
    }

    if (tid < 32) WarpReduce(data, tid);
    if (tid == 0) y[mi] = data[0];
}

void ReduceFun5(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    int block_size = CalcBlockSize(N) / 2;
    int shared_mem_size = sizeof(float) * block_size;
    ReduceKernel5<float><<<M, block_size, shared_mem_size, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}

__device__ __forceinline__ float WarpReduceUseShuffle(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16, 32);
    v += __shfl_xor_sync(0xffffffff, v, 8, 32);
    v += __shfl_xor_sync(0xffffffff, v, 4, 32);
    v += __shfl_xor_sync(0xffffffff, v, 2, 32);
    v += __shfl_xor_sync(0xffffffff, v, 1, 32);
    return v;
}

template <typename T>
__global__ void ReduceKernel6(const T *x, T *y, const int M, const int N) {
    int tid = threadIdx.x;
    int mi = blockIdx.x;
    float v = ((tid < N) ? (float)x[mi * N + tid] : 0.0f) + \
                ((tid + blockDim.x < N) ? (float)x[mi * N + tid + blockDim.x] : 0.0f);
    __syncthreads();
    auto sum = WarpReduceUseShuffle(v);
    __shared__ float buff[32];
    int wid = threadIdx.x >> 5;
    int lane_id = tid & 0x1f;
    if (lane_id == 0) buff[wid] = sum;
    __syncthreads();
    if (tid < 32) {
        v = tid < (blockDim.x >> 5) ? buff[tid] : 0.0f;
        sum = WarpReduceUseShuffle(v);
        if (tid == 0) y[mi] = sum;
    }
}

void ReduceFun6(const float* x, float* y, const int M, const int N, cudaStream_t stream) {
    int block_size = std::min((N + 1) / 2, 1024);
    block_size = (block_size + 31) / 32 * 32;
    ReduceKernel6<float><<<M, block_size, 0, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}

template <typename T>
__global__ void ReduceKernel7(const T *x, T *y, const int M, const int N) {
    int tid = threadIdx.x;
    int mi = blockIdx.x;
    float v = 0.0f;
    for (int id = tid; id < N; id += blockDim.x) v += (float)x[mi * N + id];
    __syncthreads();
    auto sum = WarpReduceUseShuffle(v);
    __shared__ float buff[32];
    int wid = threadIdx.x >> 5;
    int lane_id = tid & 0x1f;
    if (lane_id == 0) buff[wid] = sum;
    __syncthreads();
    if (tid < 32) {
        v = tid < (blockDim.x >> 5) ? buff[tid] : 0.0f;
        sum = WarpReduceUseShuffle(v);
        if (tid == 0) y[mi] = sum;
    }
}

void ReduceFun7(const float* x, float* y, const int M, const int N, cudaStream_t stream) {
    int block_size = std::min((N + 1) / 2, 1024);
    block_size = (block_size + 31) / 32 * 32;
    ReduceKernel7<float><<<M, block_size, 0, stream>>>(x, y, M, N);
    CudaRunCheck(cudaGetLastError());
}