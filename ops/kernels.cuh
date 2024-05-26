#pragma once
#include <iostream>

__global__ void VecAddKernel(const float* A, const float* B, float* C, int N);

int AddBiasRelu0(const float* x, const float* bias, float* y, const int B, const int N, cudaStream_t stream);

int AddBiasRelu1(const float* x, const float* bias, float* y, const int B, const int N, cudaStream_t stream);

int TransposeFun0(const float *x, float *y, const int M, const int N, cudaStream_t stream);

int TransposeFun1(const float *x, float *y, const int M, const int N, cudaStream_t stream);