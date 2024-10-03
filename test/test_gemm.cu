#include "use_tensor.h"
#include "logger.h"
#include "kernels.cuh"
#include <string>
#include <filesystem>
#include <functional>


std::string CUR_DIR = "/root/autodl-tmp/torchcuda"; //std::filesystem::path(__FILE__).parent_path().string();
Logger logger(CUR_DIR+"_reduce_sum_log.txt");

int TestGemm(const int M, const int N, const int K, const int method_id = 0, const int run_times = 100) {
    logger.log(LogLevel::INFO, "Matrix row: %d, col: %d, method: %d, run times: %d", M, N, method_id, run_times);
    GPUTensor<float> A({M, K}, true);
    GPUTensor<float> B({K, N}, true);
    GPUTensor<float> C({M, N}, true);
    A.random_uniform_value();
    B.random_uniform_value();
    C.fill_zero();
    std::cout << "A: " << A;
    std::cout << "B: " << B;
    auto expect_C = matmul(A, B);
    std::cout << "expect_C: " << expect_C;


    std::cout << std::endl;
    std::cout << "===== ";
    std::cout << "START method id: " << method_id;
    std::cout <<  "===== " << std::endl;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaEventBlockingSync);

    using GemmFun = 
        std::function<void(const int, const int, const int, const float *,
                    const float *, float *, cudaStream_t)>;
    std::vector<GemmFun> fun_list{gemm0, gemm1};
    std::vector<GPUTensor<float> *> io_list{&A, &B, &C};
    auto call_fun = [&](const std::vector<GPUTensor<float> *> io_list) {
        (fun_list[method_id])(M, N, K, io_list[0]->get_data("cuda"), io_list[1]->get_data("cuda"), io_list[2]->get_data("cuda"), stream);
    };

    int opts = 2 * M * N * K;
    CUDA_TIME_KERNEL_MULTIPLE(call_fun, io_list, run_times, opts);
    std::cout << "C: " << C;
    compare_GPUTensor(expect_C, C);  

    return 0;
}


int main(int argc, char **argv) {
    int M = 5;
    int N = 5;
    int K = 5;
    int run_times = 101;
    int device_id = 0;
    int fun_id = 1;
    for (int i = 1; i < argc; ++i) {
        std::string opt(argv[i]);
        if (opt == "-M")
            M = std::atoi(argv[++i]);
        else if (opt == "-N")
            N = std::atoi(argv[++i]);
        else if (opt == "-K")
            N = std::atoi(argv[++i]);
        else if (opt == "-D")
            device_id = std::atoi(argv[++i]);
        else if (opt == "-MNK")
            M = N = K = std::atoi(argv[++i]);
        else if (opt == "-R")
            run_times = std::atoi(argv[++i]);
        else if (opt == "-F")
            fun_id = std::atoi(argv[++i]);
    }
    cudaSetDevice(device_id);
    TestGemm(M, N, K, fun_id, run_times);
    return 0;
}
