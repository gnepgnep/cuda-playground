#include <stdio.h>

__global__
void HelloFromGPU()
{
    printf("********************************\n");
    printf("*        JETSON NANO           *\n");
    printf("*        PROPERTIES            *\n");
    printf("********************************\n");
}

int main ()
{
    HelloFromGPU<<<1,1>>>();
    cudaDeviceSynchronize();
    
    int deviceId;
    cudaGetDevice(&deviceId);
    
    cudaDeviceProp properties;     
    cudaGetDeviceProperties(&properties, deviceId);

    printf("Name:  %s \n", properties.name);
    printf("Multi Processor Count:  %d \n", properties.multiProcessorCount);
    printf("Max threads per block %d \n", properties.maxThreadsPerBlock);
    printf("Num of Blocks in x:  %d \n", properties.maxGridSize[0]);    
    printf("Num of Blocks in y:  %d \n", properties.maxGridSize[1]);
    printf("Num of Blocks in z:  %d \n", properties.maxGridSize[2]);
    printf("Compute Capability:  %d .%d \n", properties.major, properties.minor);

    return 0;
}