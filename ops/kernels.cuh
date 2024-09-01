#pragma once
#include <iostream>

#define CudaRunCheck(error) \
    do { \
        cudaError_t err = (error); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


void VecAddKernel(const float* A, const float* B, float* C, int N);

int AddBiasRelu0(const float* x, const float* bias, float* y, const int B, const int N, cudaStream_t stream);

int AddBiasRelu1(const float* x, const float* bias, float* y, const int B, const int N, cudaStream_t stream);

int TransposeFun0(const float *x, float *y, const int M, const int N, cudaStream_t stream);

int TransposeFun1(const float *x, float *y, const int M, const int N, cudaStream_t stream);

int TransposeFun2(const float *x, float *y, const int M, const int N, cudaStream_t stream);