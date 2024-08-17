#include "use_tensor.h"
#include "logger.h"
#include "kernels.cuh"
#include <string>
#include <filesystem>

std::string CUR_DIR = std::filesystem::path(__FILE__).parent_path().string();
Logger logger(CUR_DIR+"test_transpose_log.txt");

int TestTranspose(const int M, const int N, const int method_id = 0, const int run_times = 100) {
    logger.log(LogLevel::INFO, "Matrix row: %d, col: %d, method: %d, run times: %d", M, N, method_id, run_times);
    GPUTensor<float> x({M, N}, true);
    GPUTensor<float> y({N, M}, true);
    x.random_uniform_value();
    std::cout << "x: " << x;
    auto expect_y = x.transpose();
    std::cout << "expect_y: " << expect_y;

    std::cout << std::endl;
    std::cout << "===== ";
    std::cout << "START method id: " << method_id;
    std::cout <<  "===== " << std::endl;
    if (method_id == 0) {
        TransposeFun0(x.get_data("cuda"), y.get_data("cuda"), M, N, nullptr);
        cudaDeviceSynchronize();
        std::cout << "y: " << y;
        compare_GPUTensor(expect_y, y);
    } else if (method_id == 1) {
        TransposeFun1(x.get_data("cuda"), y.get_data("cuda"), M, N, nullptr);
        cudaDeviceSynchronize();
        std::cout << "y: " << y;
        compare_GPUTensor(expect_y, y);        
    }
    std::cout << "===== ";
    std::cout << "END method id: " << method_id;
    std::cout <<  "===== " << std::endl;
    std::cout << std::endl;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaEventBlockingSync);
    std::vector<GPUTensor<float> *> io_list{&x, &y};
    auto call_fun0 = [&](const std::vector<GPUTensor<float> *> &io_list){
        TransposeFun0(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"),
                     M, N, stream);
    };
    auto call_fun1 = [&](const std::vector<GPUTensor<float> *> &io_list){
        TransposeFun1(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"),
                     M, N, stream);
    };

    switch (method_id) {
        case 0:
            CUDA_TIME_KERNEL_MULTIPLE(call_fun0, io_list, run_times);
            break;
        case 1:
            CUDA_TIME_KERNEL_MULTIPLE(call_fun1, io_list, run_times);
            break;
    }
    return 0;
}


int main(int argc, char **argv) {
    int M = 5;
    int N = 3;
    int run_times = 11;
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
