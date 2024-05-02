#pragma once
#include <iostream>

__global__ void VecAddKernel(const float* A, const float* B, float* C, int N);

// template <typename T>
// __global__ void AddBiasRelu0Kernel(const T* x, const T* bias, T* y, const int N);

// template <typename T>
// __global__ void AddBiasReluKernel1(const T* x, const T* bias, T* y, const int N, const int Num);

int AddBiasRelu0(const float* x, const float* bias, float* y, 
                        const int B, const int N, cudaStream_t stream);

int AddBiasRelu1(const float* x, const float* bias, float* y, 
                        const int B, const int N, cudaStream_t stream);