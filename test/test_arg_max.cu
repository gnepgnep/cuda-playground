#include <iostream>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

template <typename T>
void print_vector(vector<T>& data) {
    for (auto& d: data) {
        cout << d << " ";
    }
    cout << endl;
}

template <typename T>
__device__ bool greater_func(T& data1, T& data2) {
    return data1 > data2;
}

__global__ void arg_max_kernel(int* data_d, int *max_value_d, int *max_index_d, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ int shared_max_value[128];
    __shared__ int shared_max_index[128];

    shared_max_value[tid] = idx < N ? data_d[idx]: INT_MIN;
    shared_max_index[tid] = idx < N ? idx: -1;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s>>=1) {
        if (tid < s) {
            if (greater_func(shared_max_value[tid+s], shared_max_value[tid])) {
                shared_max_value[tid] = shared_max_value[tid+s];
                shared_max_index[tid] = shared_max_index[tid+s];
            }
        } 
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_value_d, shared_max_value[0]);
        if (max_value_d[0] == shared_max_value[0]) {
            max_index_d[0] = shared_max_index[0];
        }
    }
}

__device__ void WarpArgMax(int& val, int& index) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int newval = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int newindex = __shfl_down_sync(0xFFFFFFFF, index, offset);

        if (newval > val) {
            val = newval;
            index = newindex;
        }
    }
}

__global__ void arg_max_warp_kernel(int* data_d, int *max_value_d, int *max_index_d, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    int val = (idx < N) ? data_d[tid] : INT_MIN;
    int index =  (idx < N) ? idx : -1;

    WarpArgMax(val, index);

    __shared__ int max_value[32];
    __shared__ int max_index[32];

    int wid = tid >> 5;
    int lane_id = tid & 0x1f;
    if (lane_id == 0) {
        max_value[wid] = val;
        max_index[wid] = index;
    }
    __syncthreads();

    if (tid < 32) {
        val = (tid < blockDim.x >> 5) ? max_value[tid] : INT_MIN;
        index = (tid < blockDim.x >> 5) ? max_index[tid] : -1;
        WarpArgMax(val, index);
        if (tid == 0) {
            max_value_d[0] = val;
            max_index_d[0] = index;
        }
    }


}


int main() {
    srand(41); 

    int N = 10;
    vector<int> data_h(N, 0);
    int max_value_h = INT_MIN;
    int max_index_h = -1;

    for (auto& data:data_h) {
        data = rand() % 100;
    }
    print_vector(data_h);

    int *data_d, *max_value_d, *max_index_d;

    cudaMalloc(&data_d, N*sizeof(int));
    cudaMalloc(&max_value_d, sizeof(int));
    cudaMalloc(&max_index_d, sizeof(int));

    cudaMemcpy(data_d, data_h.data(), N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_value_d, &max_value_h, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_index_d, &max_index_h, sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 128;
    int grid_size = (N + block_size - 1) / block_size;
    // arg_max_kernel<<<grid_size, block_size>>>(data_d, max_value_d, max_index_d, N);
    arg_max_warp_kernel<<<grid_size, block_size>>>(data_d, max_value_d, max_index_d, N);

    cudaMemcpy(&max_value_h, max_value_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_index_h, max_index_d, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "max value: " << max_value_h << ", index: " << max_index_h << endl;

    cudaFree(data_d);
    cudaFree(max_value_d);
    cudaFree(max_index_d);

    return 0;
}