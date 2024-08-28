#pragma once
#include "GPUTensor.cuh"

// Function to wrap around any CUDA kernel and measure its execution time for multiple iterations
#define CUDA_TIME_KERNEL_MULTIPLE(kernel_call, tensor_ptr_list, iterations)  \
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
            cudaEventRecord(stop);                                           \
                                                                             \
            cudaEventSynchronize(stop);                                      \
                                                                             \
            if (i == 0) continue;                                             \
            float milliseconds = 0;                                          \
            cudaEventElapsedTime(&milliseconds, start, stop);                \
            totalTime += milliseconds;                                       \
        }                                                                    \
                                                                             \
        float averageTime = totalTime / iterations;                          \
        printf("%s average run time %f ms over %d iterations\n",               \
                #kernel_call, averageTime, iterations);                      \
                                                                             \
        cudaEventDestroy(start);                                             \
        cudaEventDestroy(stop);                                              \
    } while(0)


#define RunStreamTrueBandwidth(fun, tensor_ptr_list, run_times, stream, mem_size_GB)\
  [&]() -> float {                                                             \
    std::vector<GpuTensor *> temp_tensor_list = tensor_ptr_list;               \
    int input_num = temp_tensor_list.size();                                   \
    std::vector<GpuTensor> copy_tensor_list(input_num *run_times * 2);         \
    std::vector<std::vector<GpuTensor *>> par_list(run_times * 2);             \
    mem_size_t allocator_size = 0;                                             \
    int real_par_list_num = 0;                                                 \
    for (std::size_t i = 0; i != par_list.size(); ++i) {                       \
      if (real_par_list_num > 0) {                                             \
        par_list[i] = par_list[i % real_par_list_num];                         \
        continue;                                                              \
      }                                                                        \
      for (int j = 0; j < input_num; ++j) {                                    \
        copy_tensor_list[i * input_num + j] =                                  \
            temp_tensor_list[j]->deep_copy();                                  \
        par_list[i].push_back(&copy_tensor_list[i * input_num + j]);           \
        allocator_size += copy_tensor_list[i * input_num + j].mem_size();      \
      }                                                                        \
      if (allocator_size > (mem_size_t)1024 * 1024 * 1024 * 2)                 \
        real_par_list_num = i + 1;                                             \
    }                                                                          \
    printf("RunStreamTrueBandwidth real par list num = %d",                    \
            real_par_list_num);                                                \
    CudaEvent start_event;                                                     \
    CudaEvent stop_event;                                                      \
    start_event.create(cudaEventCreateWithFlags, cudaEventBlockingSync);       \
    stop_event.create(cudaEventCreateWithFlags, cudaEventBlockingSync);        \
    auto cost_time = 1.0e30;                                                   \
    for (int i = 0; i < 2; ++i) {                                              \
      if (i > 1) std::this_thread::sleep_for(std::chrono::milliseconds(1000)); \
      cudaDeviceSynchronize();                                                 \
      CudaRunCheck(cudaGetLastError());                                        \
      CudaRunCheck(cudaEventRecord(start_event, stream));                      \
      for (int j = 0; j < run_times; ++j) {                                    \
        fun((i == 1 && j == run_times - 1)                                     \
                ? temp_tensor_list                                             \
                : par_list[par_list.size() - 1 - i * run_times - j]);          \
      }                                                                        \
      CudaRunCheck(cudaEventRecord(stop_event, stream));                       \
      CudaRunCheck(cudaEventSynchronize(stop_event));                          \
      float time;                                                              \
      cudaEventElapsedTime(&time, start_event, stop_event);                    \
      cost_time = time / run_times;                                            \
    }                                                                          \
    cudaDeviceSynchronize();                                                   \
    printf("%s cost time %f ms, R/W size : %fGB, Bandwidth : %fGB/s", #fun,    \
            cost_time, (float)(count), (float)(count)*1000.0f / (cost_time));  \
    return cost_time;                                                          \
  }()