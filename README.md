# torchcuda

## cuda practice project, drafting

- base: common class
    - GPUTensor: handle cpu&gpu memory allocation for tensor
    - Logger
- gpu_properties: check gpu info
- ops: cuda kernel def
    - vec_add
        - VecAddKernel (naive implementation)
    - add_bias_relu
        - AddBiasReluFun0 (naive implementation)
        - AddBiasReluFun1 (use difference indexing method)
    - transpose
        - TransposeFun0 (naive implementation)
        - TransposeFun1 (optimized memory access using shared memory)
        - TransposeFun2 (avoid shared memory bank conflict through padding)
    - reduce_sum
        - ReduceFun1 (origin implemetation)
        - ReduceFun2 (reindex tid increase working thread in each wrap)
        - ReduceFun3 (reduce bank conflict)
        - ReduceFun4 (reduce shared memory, eliminate thread that only read data)
        - ReduceFun5 (unravel the loop for the last warp avoid syncthread, forceinline avoid shared memory cache)
        - ReduceFun6 (use warp sync)
        - ReduceFun7 (reduce dim > 1024)
        - ReduceFun8 (use atomic op to get reduced result of a matrix)
    - gemm
        - gemm0 (origin implemetation)
        - gemm1 (with shared memory)
        - gemm2 (with register and compact shared mem access: using float4)
        - gemm3 (using tensor core)
        - gemm4 (increase op/mem access of gemm4)
- test: test kernel consistency and time
