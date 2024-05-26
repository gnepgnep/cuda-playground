#include "kernels.cuh"

template <typename T>
__global__ void TransposeKernel0(const T* x, T* y, const int M, const int N) {
    int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (row_idx < M && col_idx < N) {
        y[col_idx * M + row_idx] = x[row_idx * N + col_idx];
    }
}

template <typenamee T>
__global__ void TransposeKernel1(const T* x, T* y, const int M, const int N) {
    __shared__ float data[32][32];
    int row_base = blockDim.y * blockIdx.y;
    int col_base = blockDim.x * blockIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row_idx = row_base + ty;
    int col_idx = col_base + tx;
    data[ty][tx] = (row_idx < M && col_idx < N) ? x[row_idx * N + col_idx] : 0.0f;
    __syncthreads();
    row_idx = row_base + tx;
    col_idx = col_base + ty;
    if (row_idx < M && col_idx < N) {
        y[col_idx * N + row_idx] = data[tx][ty];
    }
}

int TransposeFun0(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((N+1)>>5, (M+1)>>5);
    TransposeKernel0<<<grid, block, 0, stream>>>(x, y, M, N);
    // std::cout << "cuda eror: " << cudaGetLastError() << std::endl;
    if (cudaGetLastError() != cudaSuccess) {
        printf("lauch kernel failed\n");
        return -1;
    }
    return 0;
}

int TransposeFun1(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((N+1)>>5, (M+1)>>5);
    TransposeKernel1<<<grid, block, 0, stream>>>(x, y, M, N);
    if (cudaGetLastError() != cudaSuccess) {
        printf("lauch kernel failed\n");
        return -1;
    }
    return 0;
}
