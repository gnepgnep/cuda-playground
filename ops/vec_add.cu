#include "kernels.cuh"

__global__ void VecAddKernel(const float* A, const float* B, float* C, int N){
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N){
        C[tid] = A[tid] + B[tid];
        printf("%f %f %f\n", A[tid],B[tid],C[tid]);
    }
}
