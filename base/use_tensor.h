#pragma once
#include "GPUTensor.cuh"

// Function to wrap around any CUDA kernel and measure its execution time for multiple iterations
#define CUDA_TIME_KERNEL_MULTIPLE(kernel_call, tensor_ptr_list, iterations, count)  \
    do {                                                                     \
        if (iterations < 2) {                                                \
          printf("Error: Number of iterations must be at least 2.");         \
          exit(EXIT_FAILURE);                                                \
        }                                                                    \
        cudaEvent_t start, stop;                                             \
        cudaEventCreate(&start);                                             \
        cudaEventCreate(&stop);                                              \
                                                                             \
        float totalTime = 0.0f;                                              \
        for (int i = 0; i < iterations; ++i) {                               \
            cudaEventRecord(start);                                          \
            kernel_call(tensor_ptr_list);                                    \
            cudaDeviceSynchronize();                                         \
            cudaEventRecord(stop);                                           \
                                                                             \
            cudaEventSynchronize(stop);                                      \
                                                                             \
            float milliseconds = 0;                                          \
            cudaEventElapsedTime(&milliseconds, start, stop);                \
            if (i == 0) continue;                                            \
            totalTime += milliseconds;                                       \
        }                                                                    \
                                                                             \
        float averageTime = totalTime / iterations;                          \
        printf("%s iterations %d: \ntotal run time: %fms, average run time %fms, TOPS: %f\n", \
                #kernel_call, iterations, totalTime, averageTime, (float)(count)/(averageTime*1e9));           \
                                                                             \
        cudaEventDestroy(start);                                             \
        cudaEventDestroy(stop);                                              \
    } while(0)

