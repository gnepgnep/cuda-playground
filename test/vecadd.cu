# include <iostream>
# include <vector>
# include "kernels.cuh"

int main()
{
    int N = 2000;
    std::vector<float> cpu_a(N, 0);
    std::vector<float> cpu_b(N, 0);
    std::vector<float> cpu_c(N, 0);
    for (std::size_t i=0; i != cpu_a.size(); ++i) {
        cpu_a[i] = 0.1f * i;
        cpu_b[i] = 0.2f * i;
        cpu_c[i] = cpu_a[i] + cpu_b[i];
    }
    for (std::size_t i=0; i != cpu_c.size(); ++i){
        printf("%f + %f = %f\n", cpu_a[i], cpu_b[i], cpu_c[i]);
    }
    
    float* A{nullptr};
    float* B{nullptr};
    float* C{nullptr};
    cudaMalloc(&A, sizeof(float)*N);
    cudaMalloc(&B, sizeof(float)*N);
    cudaMalloc(&C, sizeof(float)*N);
    cudaMemcpy(A, cpu_a.data(), sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B, cpu_b.data(), sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(C, cpu_c.data(), sizeof(float)*N, cudaMemcpyHostToDevice);
    dim3 grid((N+1023)/1024,1,1);
    dim3 block(1024,1,1);
    VecAddKernel<<<grid, block>>>(A, B, C, N);
    if (cudaGetLastError() != cudaSuccess){
        printf("!!! launch kernel failed\n");
        return -1;
    }
    cudaDeviceSynchronize();
    std::vector<float> cpu_c_copy(N, 0);
    cudaMemcpy(cpu_c_copy.data(), C, sizeof(float)*N, cudaMemcpyDeviceToHost);
    for (std::size_t i=0; i!=cpu_c_copy.size(); ++i){
        printf("GPU result: %f + %f = C[%ld] = %f\n", cpu_a[i], cpu_b[i], i, cpu_c_copy[i]);
    }
    return 0;
} 