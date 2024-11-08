cmake_minimum_required(VERSION 3.10)
project(TORCHCUDA LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# set c++ standard
# set(CMAKE_CXX_STANDARD 17)


# CUDA architecture configuration
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75) # Modify the architecture if needed
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Include CUDA headers
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/ops 
    ${CMAKE_CURRENT_SOURCE_DIR}/base
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include)

# Add CUDA files
file(GLOB CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test/test_gemm.cu)
file(GLOB BASE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/base/*)
file(GLOB OPS_CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cu)

# set executable output path
set(EXECUTABLE_OUTPUT_PATH ..)

# set(CMAKE_BUILD_TYPE Debug)

# Add the executable
cuda_add_executable(test_gemm ${CUDA_SOURCES} ${BASE_SOURCES} ${OPS_CUDA_SOURCES})

# Link CUDA libraries
# target_link_libraries(vecadd ${CUDA_LIBRARIES})

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# add_library(vecaddlib SHARED ${OPS_CUDA_SOURCES} ${OPS_HEADERS}) # STATIC for static lib

# static lib link
# link_libraries(vecadd)
# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# dynamic lib link
# add_exeutable(vecadd ${SRC_LIST})
# target_link_libraries(vecadd ${DYNAMIC_LIB})

# add DEBUG macro
# add_definitions(-DDEBUG)