#include "kernels.cuh"

template <typename T>
__global__ void TransposeKernel0(const T* x, T* y, const int M, const int N) {
    int row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (row_idx < M && col_idx < N) {
        y[col_idx * M + row_idx] = x[row_idx * N + col_idx];
    }
}

template <typename T>
__global__ void TransposeKernel1(const T* x, T* y, const int M, const int N) {
    __shared__ float data[32][32];
    int row_base = blockDim.y * blockIdx.y;
    int col_base = blockDim.x * blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row_idx = row_base + ty;
    int col_idx = col_base + tx;
    data[ty][tx] = (row_idx < M && col_idx < N) ? x[row_idx * N + col_idx] : 0.0f;
    __syncthreads();
    row_idx = row_base + tx;
    col_idx = col_base + ty;
    if (row_idx < M && col_idx < N) {
        y[col_idx * M + row_idx] = data[tx][ty];
    }
}

template <typename T>
__global__ void TransposeKernel2(const T* x, T* y, const int M, const int N) {
    __shared__ float data[32][33];
    int row_base = blockDim.y * blockIdx.y;
    int col_base = blockDim.x * blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row_idx = row_base + ty;
    int col_idx = col_base + tx;
    data[ty][tx] = (row_idx < M && col_idx < N) ? x[row_idx * N + col_idx] : 0.0f;
    __syncthreads();
    row_idx = row_base + tx;
    col_idx = col_base + ty;
    if (row_idx < M && col_idx < N) {
        y[col_idx * M + row_idx] = data[tx][ty];
    }
}

int TransposeFun0(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    int bdim_y = 32;
    int bdim_x = 32;
    int gdim_y = M < 32 ? 1 : (M+1)>>5;
    int gdim_x = N < 32 ? 1 : (N+1)>>5;
    dim3 block(bdim_x, bdim_y);
    dim3 grid(gdim_x, gdim_y);
    TransposeKernel0<<<grid, block, 0, stream>>>(x, y, M, N);
    // std::cout << "cuda eror: " << cudaGetLastError() << std::endl;
    CudaRunCheck(cudaGetLastError());
    if (cudaGetLastError() != cudaSuccess) {
        printf("lauch kernel failed\n");
        return -1;
    }
    return 0;
}

int TransposeFun1(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    int bdim_y = 32;
    int bdim_x = 32;
    int gdim_y = M < 32 ? 1 : (M+1)>>5;
    int gdim_x = N < 32 ? 1 : (N+1)>>5;
    dim3 block(bdim_x, bdim_y);
    dim3 grid(gdim_x, gdim_y);
    TransposeKernel1<<<grid, block, 0, stream>>>(x, y, M, N);
    if (cudaGetLastError() != cudaSuccess) {
        printf("lauch kernel failed\n");
        return -1;
    }
    return 0;
}

int TransposeFun2(const float *x, float *y, const int M, const int N, cudaStream_t stream) {
    int bdim_y = 32;
    int bdim_x = 32;
    int gdim_y = M < 32 ? 1 : (M+1)>>5;
    int gdim_x = N < 32 ? 1 : (N+1)>>5;
    dim3 block(bdim_x, bdim_y);
    dim3 grid(gdim_x, gdim_y);
    TransposeKernel2<<<grid, block, 0, stream>>>(x, y, M, N);
    if (cudaGetLastError() != cudaSuccess) {
        printf("lauch kernel failed\n");
        return -1;
    }
    return 0;
}
