/*
gemm0: origin implemetation
gemm1: with shared memory
gemm2: with register and compact shared mem access: using float4
gemm3: using tensor core
gemm4: increase op/mem access of gemm4
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


template <typename T, int BX = 16, int BY = 16, int TILE_M = 64, int TILE_N = 64, int TILE_K = 16>
__global__ void GemmKernel2(
    const int M, const int N, const int K, 
    T *__restrict__ C, const T *__restrict__ A, const T *__restrict__ B) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m_offset = blockIdx.y * TILE_M;
    int n_offset = blockIdx.x * TILE_N;
    int k_offset = 0;

    __shared__ float4 lbuff[BY][TILE_K], rbuff[TILE_K][BX];
    float reg[4][4] = {0.0f};
    float lreg[4], rreg[4];

    while (k_offset < K) {
        if (m_offset + TILE_M <= M && n_offset + TILE_N <= N && k_offset + TILE_K <= K) {
            // load A
            auto cur_A = A + ((m_offset + ty) * K + k_offset + tx);
            lbuff[ty][tx].x = (float)(*cur_A);
            cur_A += BY * K;
            lbuff[ty][tx].y = (float)(*cur_A);
            cur_A += BY * K;
            lbuff[ty][tx].z = (float)(*cur_A);
            cur_A += BY * K;
            lbuff[ty][tx].w = (float)(*cur_A);

            // load B
            auto cur_B = B + ((k_offset + ty) * N + n_offset + tx);
            rbuff[ty][tx].x = (float)(*cur_B);
            cur_B += BX;
            rbuff[ty][tx].y = (float)(*cur_B);
            cur_B += BX;
            rbuff[ty][tx].z = (float)(*cur_B);
            cur_B += BX;
            rbuff[ty][tx].w = (float)(*cur_B);
        } else {
        }
        __syncthreads();

        for (int ki = 0; ki < TILE_K; ++ki) {
            *(reinterpret_cast<float4 *>(lreg)) = lbuff[ty][ki];
            *(reinterpret_cast<float4 *>(rreg)) = rbuff[ki][tx];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    reg[i][j] += lreg[i] * rreg[j];
                }
            }
        }
        __syncthreads();

        k_offset += TILE_K;
    }
    if (m_offset + TILE_M <= M && n_offset + TILE_N <= N) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int mi = m_offset + i * BY + ty;
                int ni = n_offset + j * BX + tx;
                C[mi * N + ni] = reg[i][j];
            }
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int mi = m_offset + i * BY + ty;
                int ni = n_offset + j * BX + tx;
                if (mi < M && ni < N) {
                    C[mi * N + ni] = reg[i][j];
                }
            }
        }        
    }
}

void gemm2(const int M, const int N, const int K, const float* A, const float* B, float* C, cudaStream_t stream) {
    constexpr int BXY = 16;
    constexpr int TILE = 64;
    dim3 block(BXY, BXY, 1);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    GemmKernel2<float, BXY, BXY, TILE, TILE, BXY><<<grid, block, 0, stream>>>(M, N, K, C, A, B);
    CudaRunCheck(cudaGetLastError());
}