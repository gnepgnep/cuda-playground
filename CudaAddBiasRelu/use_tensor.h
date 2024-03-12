#pragma once
#include "base/GPUTensor.h"

using namespace base;

#define CudaRunCheck(error) \
    do { \
        cudaError_t err = (error); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#define RunStreamTrueBandwidth(fun, tensor_ptr_list, run_times, stream, mem_size_GB) \
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
    return cost_time;
  }()