#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

// Test configuration with tb_k=64 and warp_k=64
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
        std::cout << "FAILED (can_implement returned " << int(status) << ")" << std::endl;
        return false;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "FAILED (initialize returned " << int(status) << ")" << std::endl;
        return false;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cout << "FAILED (execution returned " << int(status) << ")" << std::endl;
        return false;
    }

    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        std::cout << "FAILED (CUDA error: " << cudaGetErrorString(cuda_status) << ")" << std::endl;
        return false;
    }

    std::cout << "SUCCESS" << std::endl;
    return true;
}

int main() {
    const int M = 2048, N = 2048, K = 2048;

    std::cout << "Testing TF32 CUTLASS kernels with different K dimensions" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Instruction shape: 16x8x8 (TF32)" << std::endl;
    std::cout << std::endl;

    // Test with tb_k=32, warp_k=32 (baseline)
    using Config32_32_32 = MatMulConfig<64, 64, 32, 32, 32, 32, 2>;
    test_config<typename Config32_32_32::Gemm>("TB:64x64x32, Warp:32x32x32, Stages:2", M, N, K);

    // Test with tb_k=64, warp_k=32 (tb_k=64, but warp still 32)
    using Config64_32_32 = MatMulConfig<64, 64, 64, 32, 32, 32, 2>;
    test_config<typename Config64_32_32::Gemm>("TB:64x64x64, Warp:32x32x32, Stages:2", M, N, K);

    // Test with tb_k=64, warp_k=64 (both 64)
    using Config64_64_64 = MatMulConfig<64, 64, 64, 64, 32, 64, 2>;
    test_config<typename Config64_64_64::Gemm>("TB:64x64x64, Warp:64x32x64, Stages:2", M, N, K);

    // Test with tb_k=32, warp_k=16 (smaller K)
    using Config32_32_16 = MatMulConfig<64, 64, 32, 32, 32, 16, 2>;
    test_config<typename Config32_32_16::Gemm>("TB:64x64x32, Warp:32x32x16, Stages:2", M, N, K);

    // Test with tb_k=64, warp_k=16
    using Config64_32_16 = MatMulConfig<64, 64, 64, 32, 32, 16, 2>;
    test_config<typename Config64_32_16::Gemm>("TB:64x64x64, Warp:32x32x16, Stages:2", M, N, K);

    std::cout << std::endl;
    std::cout << "Note: The instruction shape K=8 means warp_k must be multiple of 8" << std::endl;
    std::cout << "Valid warp_k values: 8, 16, 24, 32, 40, 48, 56, 64, ..." << std::endl;

    return 0;
}
