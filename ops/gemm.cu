/*
gemm0: origin implemetation
gemm1: with shared memory
gemm2: with register, increase op/mem access
gemm3: with compact shared mem access: using float4
gemm4: using tensor core
gemm5: increase op/mem access of gemm4
*/
#include "kernels.cuh"

template <typename T>
__global__ void GemmKernel0(const int M, const int N, const int K, T* C, const T* A, const T* B) {
    int mi = blockDim.y * blockIdx.y + threadIdx.y;
    int ni = blockDim.x * blockIdx.x + threadIdx.x;
    float temp = 0.0;
    for (int ki = 0; ki < K; ki++) {
        temp += (float)A[mi * K + ki] * B[ki * N + ni];
    }
    C[mi * N + ni] = (T)temp;
}


void gemm0(const int M, const int N, const int K, const float* A, const float* B, float* C, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32 , (M + 31) / 32);
    GemmKernel0<<<grid, block, 0, stream>>>(M, N, K, C, A, B);
    CudaRunCheck(cudaGetLastError());
}