# include <iostream>
# include <vector>

__global__ void VecAddKernel(const float* A, const float* B, float* C){
    auto tid = threadIdx.x;
    C[tid] = A[tid] + B[tid];
}

int main()
{
    int N = 100;
    std::vector<float> cpu_a(N, 0);
    std::vector<float> cpu_b(N, 0);
    std::vector<float> cpu_c(N, 0);
    for (std::size_t i=0; i != cpu_a.size(); ++i) {
        cpu_a[i] = 0.1f * i;
        cpu_b[i] = 0.2f * i;
        cpu_c[i] = cpu_a[i] + cpu_b[i];
    }
    for (std::size_t i=0; i != cpu_c.size(); ++i){
        printf("%f + %f = %f\n", cpu_a[i], cpu_c[i], cpu_a[i]);
    }
    
    float* A{nullptr};
    float* B{nullptr};
    float* C{nullptr};
    cudaMemcpy(A, cpu_a.data(), sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B, cpu_b.data(), sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(C, cpu_c.data(), sizeof(float)*N, cudaMemcpyHostToDevice);
    dim3 grid(1,1,1);
    dim3 block(N,1,1);
    VecAddKernel<<<grid, block>>>(A, B, C);
    cudaDeviceSynchronize();
    std::vector<float> cpu_c_copy(N, 0);
    cudaMemcpy(cpu_c_copy.data(), C, sizeof(float)*N, cudaMemcpyDefault);
    for (std::size_t i=0; i!=cpu_c_copy.size(); ++i){
        printf("GPU result: C[%d] = %f\n", i, cpu_c_copy[i]);
    }
    return 0;
}