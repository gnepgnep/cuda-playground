/*****************************************************************
 * c/c++ head file
 * encoding: utf-8
 *     Author : wangzw
 *     Created Time: Thu 15 Oct 2020 06:43:56 PM CST
 *     Last Change : 2021年10月29日 星期五 15时04分09秒
 *******************************************************************/
#pragma once
#include <thread>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include "base/log.h"
#include "base/cuda_base.h"
#include "base/time.h"
#include "core/any_tensor.h"
#include "arithmetic/tensor_arithmetic.h"
#include "core/tensor_util.h"
#include "core/allocator.h"
#include "cuda/cuda_type_util.h"
#include "cudnn/cu_handle.h"
#include "cudnn/cuda_type.h"
#include "cudnn/softmax_fun.h"
#if HAVE_PANTHER == 1
#include "panther/panther_types.h"
#include "panther/math/panther_math.h"
#include "panther/panther_params.h"
#endif
using namespace byte::core;
using namespace byte::base;
using namespace byte::arithmetic;
// using namespace byte::cudnn;
#include <functional>
GpuTensor Fp16Matmul(const GpuTensor &a, const GpuTensor &b,
                     const int fp16_mode) {
  int data_type = a.data_type();
  if (fp16_mode == 1) {
    return matmul(a.cast(D_FLOAT16).cast(data_type),
                  b.cast(D_FLOAT16).cast(data_type));
  } else if (fp16_mode == 2) {
    return matmul(a.cast(D_FLOAT16), b.cast(D_FLOAT16)).cast(data_type);
  } else {
    return matmul(a, b);
  }
}
using CudaStream = byte::cuda::CudaType<cudaStream_t, cudaError_t, cudaSuccess,
                                        cudaStreamDestroy, cudaStreamCreate>;
inline float GetInt8Scale(GpuTensor &t) {
  cudaDeviceSynchronize();
  auto data_type = t.data_type();
  auto scale = t.cast(D_FLOAT32).abs().max().cpu_copy().float_value(0) / 127.0;
  t = ((t.cast(D_FLOAT32) / scale).cast(D_INT8).cast(D_FLOAT32) * scale)
          .cast(data_type);
  return scale;
}
inline cudaStream_t GetDefaultStream() {
  static std::unique_ptr<CudaStream> stream;
  if (stream == nullptr) {
    stream.reset(new CudaStream(false));
    stream->create(cudaStreamCreateWithFlags, cudaEventBlockingSync);
  }
  return stream->get();
}
template <class Fun>
GpuTensor ComputeGrad(GpuTensor &x, Fun &forward_fun,
                      const float deta = 0.001f) {
  GpuTensor grad_x(x.data_type(), x.shape(), true);
  for (auto i = 0; i < x.size(); ++i) {
    auto ori_v = x[i];
    x.set_pos_value(i, ori_v + deta);
    auto f1 = forward_fun(x);
    x.set_pos_value(i, ori_v - deta);
    auto f2 = forward_fun(x);
    cudaDeviceSynchronize();
    x.set_pos_value(i, ori_v);
    auto g = (f1 - f2) / (2.0f * deta);
    grad_x.set_pos_value(i, g);
  }
  return std::move(grad_x);
}
using CudaEvent = ::byte::cuda::CudaType<cudaEvent_t, cudaError, cudaSuccess,
                                         cudaEventDestroy, cudaEventCreate>;
#if DEBUG_LOG_MODE == 1
#define RunStreamPerf(run_op, run_times, stream)                    \
  [&]() -> float {                                                  \
    LogInfo("RunStreamPerf, call fun %s", #run_op);                 \
    cudaDeviceSynchronize();                                        \
    for (int i = 0; i < std::min<int>(run_times, 1); ++i) (run_op); \
    cudaStreamSynchronize(stream);                                  \
    cudaDeviceSynchronize();                                        \
    return 0.0f;                                                    \
  }()
#else
#define RunStreamPerf(run_op, run_times, stream)                               \
  [&]() -> float {                                                             \
    CudaEvent start_event;                                                     \
    CudaEvent stop_event;                                                      \
    start_event.create(cudaEventCreateWithFlags, cudaEventBlockingSync);       \
    stop_event.create(cudaEventCreateWithFlags, cudaEventBlockingSync);        \
    auto cost_time = 0.0f;                                                     \
    for (int i = 0; i < 2; ++i) {                                              \
      if (i > 1) std::this_thread::sleep_for(std::chrono::milliseconds(1000)); \
      cudaDeviceSynchronize();                                                 \
      CudaRunCheck(cudaGetLastError());                                        \
      CudaRunCheck(cudaEventRecord(start_event, stream));                      \
      for (int j = 0; j < run_times; ++j) {                                    \
        (run_op);                                                              \
      }                                                                        \
      CudaRunCheck(cudaEventRecord(stop_event, stream));                       \
      CudaRunCheck(cudaEventSynchronize(stop_event));                          \
      float time;                                                              \
      cudaEventElapsedTime(&time, start_event, stop_event);                    \
      cost_time = time / run_times;                                            \
    }                                                                          \
    cudaDeviceSynchronize();                                                   \
    LogInfo("%s cost time %f ms", #run_op, cost_time);                         \
    return cost_time;                                                          \
  }()
#endif
#define RunStreamTops(run_op, run_times, stream, count)                        \
  [&]() -> float {                                                             \
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
        (run_op);                                                              \
      }                                                                        \
      CudaRunCheck(cudaEventRecord(stop_event, stream));                       \
      CudaRunCheck(cudaEventSynchronize(stop_event));                          \
      float time;                                                              \
      cudaEventElapsedTime(&time, start_event, stop_event);                    \
      cost_time = time / run_times;                                            \
    }                                                                          \
    cudaDeviceSynchronize();                                                   \
    LogInfo("%s cost time %f ms, TOPS : %f", #run_op, cost_time,               \
            (float)count / (cost_time * 1e9));                                 \
    return cost_time;                                                          \
  }()

#define RunStreamBandwidth(run_op, run_times, stream, count)                   \
  [&]() -> float {                                                             \
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
        (run_op);                                                              \
      }                                                                        \
      CudaRunCheck(cudaEventRecord(stop_event, stream));                       \
      CudaRunCheck(cudaEventSynchronize(stop_event));                          \
      float time;                                                              \
      cudaEventElapsedTime(&time, start_event, stop_event);                    \
      cost_time = time / run_times;                                            \
    }                                                                          \
    cudaDeviceSynchronize();                                                   \
    LogInfo("%s cost time %f ms, Bandwidth : %f GB/s", #run_op, cost_time,     \
            (float)count * 1000.0f / cost_time);                               \
    return cost_time;                                                          \
  }()

#if DEBUG_LOG_MODE == 1
#define RunStreamTrueTops(fun, tensor_ptr_list, run_times, stream, count) \
  [&]() -> float {                                                        \
    LogInfo("RunStreamTrueTops, call fun %s(%s), opts : %d", #fun,        \
            #tensor_ptr_list, count);                                     \
    cudaDeviceSynchronize();                                              \
    for (int i = 0; i < std::min<int>(run_times, 1); ++i)                 \
      fun(tensor_ptr_list);                                               \
    cudaStreamSynchronize(stream);                                        \
    cudaDeviceSynchronize();                                              \
    return 0.0f;                                                          \
  }()
#else
#define RunStreamTrueTops(fun, tensor_ptr_list, run_times, stream, count)      \
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
    LogInfo("%s cost time %f ms, TOPS : %f", #fun, cost_time,                  \
            (float)(count) / (cost_time * 1e9));                               \
    return cost_time;                                                          \
  }()
#endif

#if DEBUG_LOG_MODE == 1
#define RunStreamTrueBandwidth(fun, tensor_ptr_list, run_times, stream, count) \
  [&]() -> float {                                                             \
    (void)(count);                                                             \
    LogInfo("RunStreamBandwidth, call fun %s(%s)", #fun, #tensor_ptr_list);    \
    cudaDeviceSynchronize();                                                   \
    for (int i = 0; i < std::min<int>(run_times, 1); ++i)                      \
      fun(tensor_ptr_list);                                                    \
    cudaStreamSynchronize(stream);                                             \
    cudaDeviceSynchronize();                                                   \
    return 0.0f;                                                               \
  }()
#else
#define RunStreamTrueBandwidth(fun, tensor_ptr_list, run_times, stream, count) \
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
    LogInfo("RunStreamTrueBandwidth real par list num = %d",                   \
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
    LogInfo("%s cost time %f ms, R/W size : %fGB, Bandwidth : %fGB/s", #fun,   \
            cost_time, (float)(count), (float)(count)*1000.0f / (cost_time));  \
    return cost_time;                                                          \
  }()
#endif
#define RunCheckPerf(run_op, run_times)                                        \
  {                                                                            \
    auto cost_time = 1.0e30;                                                   \
    for (int i = 0; i < 2; ++i) {                                              \
      if (i > 1) std::this_thread::sleep_for(std::chrono::milliseconds(1000)); \
      cudaDeviceSynchronize();                                                 \
      auto start = GetMilliTimef();                                            \
      for (int j = 0; j < run_times; ++j) {                                    \
        (run_op);                                                              \
      }                                                                        \
      cudaDeviceSynchronize();                                                 \
      auto end = GetMilliTimef();                                              \
      cost_time = (end - start) / run_times;                                   \
    }                                                                          \
    LogInfo("%s cost time %f ms", #run_op, cost_time);                         \
  }

#define RunCheckTops(exp, run_times, op_times)                      \
  {                                                                 \
    auto cost_time = 0.0f;                                          \
    for (int i = 0; i < 2; ++i) {                                   \
      std::this_thread::sleep_for(std::chrono::milliseconds(1000)); \
      cudaDeviceSynchronize();                                      \
      auto start = GetMilliTimef();                                 \
      for (int j = 0; j < run_times; ++j) {                         \
        (exp);                                                      \
      }                                                             \
      cudaDeviceSynchronize();                                      \
      auto end = GetMilliTimef();                                   \
      cost_time = (end - start) / run_times;                        \
    }                                                               \
    LogInfo("%s cost time %f ms, TOPS : %f", #exp, cost_time,       \
            (float)(op_times) / (cost_time * 1e9));                 \
  }
inline void AssignCShape(int *c_shape, const TensorShape &shape) {
  for (int i = 0; i < shape.dims(); ++i) {
    c_shape[i] = shape[i];
  }
}
