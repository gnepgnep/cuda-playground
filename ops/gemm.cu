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


template <typename T, int TILE_M = 32, int TILE_N = 32, int TILE_K = 32>
__global__ void GemmKernel1(
    const int M, const int N, const int K, 
    T *__restrict__ C, const T *__restrict__ A, const T *__restrict__ B) {
    __shared__ float lbuff[TILE_M][TILE_K], rbuff[TILE_K][TILE_N];
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int n_offset = blockIdx.x * TILE_N;
    int m_offset = blockIdx.y * TILE_M;
    int k_offset = 0;
    float sum = 0.0f;
    while (k_offset < K) {
        // load A
        for (auto id = tx; id < TILE_K; id += blockDim.x) {
            lbuff[ty][id] = (m_offset + ty < M && id + k_offset < K) 
                            ? (float)A[(m_offset + ty) * K + id + k_offset] : 0.0f;
        }
        // load B
        for (auto id = ty; id < TILE_K; id += blockDim.y) {
            rbuff[id][tx] = (n_offset + tx < N && id + k_offset < K)
                            ? (float)B[(id + k_offset) * N + n_offset + tx] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int ki = 0; ki < TILE_K; ++ki) {
            sum += lbuff[ty][ki] * rbuff[ki][tx];
        }
        __syncthreads();
        k_offset += TILE_K;
    }
    if (m_offset + ty < M && n_offset + tx < N) {
        C[(m_offset + ty) * N + n_offset + tx] = (T)sum;
    }

}


void gemm1(const int M, const int N, const int K, const float* A, const float* B, float* C, cudaStream_t stream) {
    dim3 block(32,32,1);
    dim3 grid((N+31) / 32, (M+31) / 32);
    GemmKernel1<float,32,32,32><<<grid, block, 0, stream>>>(M, N, K, C, A, B);
    CudaRunCheck(cudaGetLastError());
}