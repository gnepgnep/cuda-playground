#include "use_tensor.h"
#include "logger.h"
#include "kernels.cuh"
#include <string>
#include <filesystem>

std::string CUR_DIR = std::filesystem::path(__FILE__).parent_path().string();
Logger logger(CUR_DIR+"_reduce_sum_log.txt");

int TestReduce(const int M, const int N, const int method_id = 0, const int run_times = 100) {
    logger.log(LogLevel::INFO, "Matrix row: %d, col: %d, method: %d, run times: %d", M, N, method_id, run_times);
    GPUTensor<float> x({M, N}, true);
    GPUTensor<float> y({M}, true);
    x.random_uniform_value();
    std::cout << "x: " << x;
    auto expect_y = x.reduce_sum();
    std::cout << "expect_y: " << expect_y;

    std::cout << std::endl;
    std::cout << "===== ";
    std::cout << "START method id: " << method_id;
    std::cout <<  "===== " << std::endl;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaEventBlockingSync);
    auto call_fun = [&](const std::vector<GPUTensor<float> *> io_list) -> int {
        switch (method_id){
            case 1:
                ReduceFun1(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"), M, N, stream);
                break;
            case 2:
                ReduceFun2(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"), M, N, stream);
                break;
            case 3:
                ReduceFun3(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"), M, N, stream);
                break;
            case 4:
                ReduceFun4(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"), M, N, stream);
                break;
            case 5:
                ReduceFun5(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"), M, N, stream);
                break;
            case 6:
                ReduceFun6(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"), M, N, stream);
                break;
            case 7:
                ReduceFun7(io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"), M, N, stream);
                break;
            default:
                logger.log(LogLevel::ERROR, "unsupport method id: %d", method_id);
                return -1;
        }
        return 0;
    };

    std::vector<GPUTensor<float> *> io_list{&x, &y};
    int opts = M * N;
    CUDA_TIME_KERNEL_MULTIPLE(call_fun, io_list, run_times, opts);
    std::cout << "y: " << y;
    compare_GPUTensor(expect_y, y);  

    return 0;
}


int main(int argc, char **argv) {
    int M = 1024;
    int N = 1024;
    int run_times = 101;
    int device_id = 0;
    int fun_id = 1;
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
    if (fun_id >= 1) {
        TestReduce(M, N, fun_id, run_times);
    }
    return 0;
}
