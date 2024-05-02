# Makefile for vecadd.cu

# Compiler
NVCC := /usr/local/cuda/bin/nvcc

# Directories
CUDA_INCLUDE := -I/usr/local/cuda/include
CUDA_LIB := -L/usr/local/cuda/lib64

# Flags
NVCC_FLAGS := -gencode arch=compute_75,code=sm_75

# Target
TARGET := vecadd

# Compile and link
$(TARGET): vecadd.cu
	$(NVCC) $(CUDA_INCLUDE) $(CUDA_LIB) $(NVCC_FLAGS) $< -o $@ -lcudart

# Clean
clean:
	rm -f $(TARGET)
