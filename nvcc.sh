/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart \
-gencode arch=compute_75,code=sm_75 cuda_test/cuda_test.cu -o cuda_test/cuda_test