#include "use_tensor.h"
#include "logger.h"
#include "kernels.cuh"
#include <string>
#include <filesystem>

std::string CUR_DIR = std::filesystem::path(__FILE__).parent_path().string();
Logger logger(CUR_DIR+"test_transpose_log.txt");

int TestTranspose(const int M, const int N, const int method_id = 0, const int run_times = 100) {
    logger.log(LogLevel::INFO, "Matrix row: %d, col: %d, method: %d, run times: %d", M, N, method_id, run_times);
    GPUTensor x(D_FLOAT32, {M, N}, true);
    GPUTensor y(D_FLOAT32, {N, M}, true);
    x.random_uniform_value();
    std::cout << "x: " << x;

    auto expect_y = x.transpose();

    std::cout << "expect_y: " << expect_y;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaEventBlockingSync);
    TransposeFun0(x.get_data<float>("cuda"), y.get_data<float>("cuda"), M, N, stream);
    cudaDeviceSynchronize();
    std::cout << "y: " << y;
    compare_GPUTensor(expect_y, y);
}


int main(int argc, char **argv) {
    int M = 3;
    int N = 5;
    int run_times = 100;
    int device_id = 0;
    int fun_id = -1;
    for (int i = 1; i < argc; ++i) {
        std::string opt(argv[i]);
        if (opt == "-M")
            M = std::atoi(argv[++i]);
        else if (opt == "-N")
            N = std::atoi(argv[++i]);
        else if (opt == "-R")
            run_times = std::atoi(argv[++i]);
        else if (opt == "-f")
            fun_id = std::atoi(argv[++i]);
    }
    cudaSetDevice(device_id);
    if (fun_id >= 0) {
        TestTranspose(M, N, fun_id, run_times);
    } else {
        TestTranspose(M, N, 0, run_times);
        TestTranspose(M, N, 1, run_times);
        TestTranspose(M, N, 2, run_times);
    }
}
