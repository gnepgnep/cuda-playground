/*
gemm0: origin implemetation
gemm1: with shared memory
gemm2: with register and compact shared mem access: using float4
gemm3: using tensor core
gemm4: increase op/mem access of gemm4
*/
#include "kernels.cuh"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

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

template <typename T, int TILE_M, int TILE_N, int TILE_K>
struct ABShared {
    T lbuff[TILE_M][TILE_K];
    T rbuff[TILE_K][TILE_N];
};

template <typename T, int TILE_M, int TILE_N>
struct CShared {
    T cbuff[TILE_M][TILE_N];
};

extern __shared__ int8_t shared_buff[];
template <typename T, int WM = 16, int WN = 16, int WK = 16, int WarpM = 4, int WarpN = 4, int TILE_M = WM * WarpM, int TILE_N = WN * WarpN, int TILE_K = WK>
__global__ void GemmKernel3(
    const int M, const int N, const int K, 
    T *__restrict__ C, const T *__restrict__ A, const T *__restrict__ B) {

    using ABSharedType = ABShared<half, TILE_M, TILE_N, TILE_K>;
    using CSharedType = CShared<float, TILE_M, TILE_N>;
    ABSharedType *ab_shared = reinterpret_cast<ABSharedType *>(shared_buff);
    CSharedType *c_shared = reinterpret_cast<CSharedType *>(shared_buff);
    auto &lbuff = ab_shared->lbuff;
    auto &rbuff = ab_shared->rbuff;
    auto &cbuff = c_shared->cbuff;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int warp_mid = warp_id / WarpN;
    int warp_nid = warp_id - warp_mid * WarpN;

    int m_offset = blockIdx.y * TILE_M;
    int n_offset = blockIdx.x * TILE_N;
    int k_offset = 0;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    while (k_offset < K) {
        if (m_offset + TILE_M <= M && n_offset + TILE_N <= N && k_offset + TILE_K <= K) {
            // load A
            for (int id = tid; id < TILE_K * TILE_M; id += blockDim.x) {
                int m = id / TILE_K;
                int k = id % TILE_K;
                lbuff[m][k] = (half)A[(m_offset + m) * K + k_offset + k];
            }
            // load B
            for (int id = tid; id < TILE_K * TILE_N; id += blockDim.x) {
                int n = id / TILE_K;
                int k = id % TILE_K;
                rbuff[k][n] = (half)B[(k_offset + k) * N + n_offset + n];
            }
        } else {
            // load A
            for (int id = tid; id < TILE_K * TILE_M; id += blockDim.x) {
                int m = id / TILE_K;
                int k = id % TILE_K;
                lbuff[m][k] = (m_offset + m < M && k_offset + k < K) ? (half)A[(m_offset + m) * K + k_offset + k] :  (half)0.0f;
            }
            // load B
            for (int id = tid; id < TILE_K * TILE_N; id += blockDim.x) {
                int n = id / TILE_K;
                int k = id % TILE_K;
                rbuff[k][n] = (n_offset + n < N && k_offset + k < K) ? (half)B[(k_offset + k) * N + n_offset + n] : (half)0.0f;
            }
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, (half*)lbuff + (warp_mid * WM) * TILE_K, TILE_K);
        wmma::load_matrix_sync(b_frag, (half*)rbuff + warp_nid * WN, TILE_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        k_offset += TILE_K;
        __syncthreads();
    }
    wmma::store_matrix_sync((float*)cbuff + (warp_mid * WN) * TILE_N + warp_nid * WN, acc_frag, TILE_N, wmma::mem_row_major);
    __syncthreads();
    
    auto cur_C = C + m_offset * N + n_offset;
    if (m_offset + TILE_M <= M && n_offset + TILE_N <= N) {
        for (int id = tid; id < TILE_M * TILE_N; id += blockDim.x) {
            int m = id / TILE_N;
            int n = id % TILE_N;
            cur_C[m * N + n] = (T)cbuff[m][n];
        }
    } else {
        for (int id = tid; id < TILE_M * TILE_N; id += blockDim.x) {
            int m = id / TILE_N;
            int n = id % TILE_N;
            cur_C[m * N + n] = (T)cbuff[m][n];
        }
    }
}


void gemm3(const int M, const int N, const int K, const float* A, const float* B, float* C, cudaStream_t stream) {
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int WK = 16;
    constexpr int WarpM = 4;
    constexpr int WarpN = 4;
    constexpr int TILE_M = WM * WarpM;
    constexpr int TILE_N = WN * WarpN;
    constexpr int TILE_K = WK;
    int ab_shared_size = sizeof(half) * (TILE_M + TILE_K) * (TILE_N + TILE_K);
    int c_shared_size = sizeof(float) * TILE_M * TILE_N;
    int shared_size = std::max(ab_shared_size, c_shared_size);
    if (shared_size > 48 * 1024) {
        printf("shared memory size: %d KB, exceeds the limit: 48 KB\n", shared_size / 1024);
        exit(EXIT_FAILURE);
    }
    int block_size = WarpM * WarpN * 32;
    dim3 grid((N+TILE_N-1) / TILE_N, (M+TILE_M-1) / TILE_M);
    GemmKernel3<float, WM, WN, WK, WarpM, WarpN, TILE_M, TILE_N, TILE_K><<<grid, block_size, shared_size, stream>>>(M, N, K, C, A, B);
    CudaRunCheck(cudaGetLastError());
}

