#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

// Test different K values for TF32
template<int TBM, int TBN, int TBK, int WM, int WN, int WK, int S>
struct MatMulConfig {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;  // TF32
    static constexpr int kStages = S;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
        ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;
};

template<typename GemmKernel>
bool test_config(const char* config_name, int M, int N, int K) {
    std::cout << "Testing " << config_name << "... ";

    using ElementA = typename GemmKernel::ElementA;
    using ElementB = typename GemmKernel::ElementB;
    using ElementC = typename GemmKernel::ElementC;

    cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<ElementB, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C({M, N});

    for (int i = 0; i < M * K; i++) A.host_data()[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B.host_data()[i] = 1.0f;
    for (int i = 0; i < M * N; i++) C.host_data()[i] = 0.0f;
    A.sync_device();
    B.sync_device();
    C.sync_device();

    float alpha = 1.0f, beta = 0.0f;
    typename GemmKernel::Arguments args{
        {M, N, K}, {A.device_ref()}, {B.device_ref()},
        {C.device_ref()}, {C.device_ref()}, {alpha, beta}};

    GemmKernel gemm_op;

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "FAILED (can_implement)" << std::endl;
        return false;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "FAILED (initialize)" << std::endl;
        return false;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cout << "FAILED (execution)" << std::endl;
        return false;
    }

    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        std::cout << "FAILED (CUDA)" << std::endl;
        return false;
    }

    std::cout << "SUCCESS" << std::endl;
    return true;
}

int main() {
    const int M = 2048, N = 2048, K = 2048;

    std::cout << "Testing TF32 CUTLASS with different K dimensions" << std::endl;
    std::cout << "Instruction shape: 16x8x8 (K=8)" << std::endl;
    std::cout << "Threadblock: 64x64xK, Warp: 32x32xK, Stages: 2" << std::endl;
    std::cout << std::endl;

    // Test K=16 (2× instruction K)
    using Config_K16 = MatMulConfig<64, 64, 16, 32, 32, 16, 2>;
    test_config<typename Config_K16::Gemm>("tb_k=16, warp_k=16", M, N, K);

    // Test K=24 (3× instruction K)
    using Config_K24 = MatMulConfig<64, 64, 24, 32, 32, 24, 2>;
    test_config<typename Config_K24::Gemm>("tb_k=24, warp_k=24", M, N, K);

    // Test K=32 (4× instruction K) - BASELINE
    using Config_K32 = MatMulConfig<64, 64, 32, 32, 32, 32, 2>;
    test_config<typename Config_K32::Gemm>("tb_k=32, warp_k=32", M, N, K);

    // Test K=40 (5× instruction K)
    using Config_K40 = MatMulConfig<64, 64, 40, 32, 32, 40, 2>;
    test_config<typename Config_K40::Gemm>("tb_k=40, warp_k=40", M, N, K);

    // Test K=48 (6× instruction K)
    using Config_K48 = MatMulConfig<64, 64, 48, 32, 32, 48, 2>;
    test_config<typename Config_K48::Gemm>("tb_k=48, warp_k=48", M, N, K);

    // Test K=56 (7× instruction K)
    using Config_K56 = MatMulConfig<64, 64, 56, 32, 32, 56, 2>;
    test_config<typename Config_K56::Gemm>("tb_k=56, warp_k=56", M, N, K);

    std::cout << std::endl;
    std::cout << "Summary: Testing which K values work for TF32 (instruction K=8)" << std::endl;

    return 0;
}
