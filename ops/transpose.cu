#include "kernels.cuh"

template <typename T>
__global__ void TransposeKernel0(const T* x, T* y, const int M, const int N) {
    int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (row_idx < M && col_idx < N) {
        y[col_idx * M + row_idx] = x[row_idx * N + col_idx];
    }
}

int TransposeFun0(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((N+1)>>5, (M+1)>>5);
    TransposeKernel0<<<grid, block, 0, stream>>>(x, y, M, N);
    if (cudaGetLastError() != cudaSuccess) {
        printf("lauch kernel failed\n");
        return -1;
    }
    return 0;
}