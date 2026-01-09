#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

// Test extreme K values: K=8 and K=128
template<int TBK, int WK>
struct MatMulConfig {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, TBK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;  // TF32
    static constexpr int kStages = 2;
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
bool test_kernel(const char* name, int M, int N, int K) {
    printf("%-35s ... ", name);
    fflush(stdout);

    cutlass::HostTensor<float, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C({M, N});

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
        printf("FAILED (can_implement)\n");
        return false;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        printf("FAILED (initialize)\n");
        return false;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        printf("FAILED (execute)\n");
        return false;
    }

    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        printf("FAILED (CUDA: %s)\n", cudaGetErrorString(cuda_status));
        return false;
    }

    printf("SUCCESS\n");
    return true;
}

int main() {
    const int M = 2048, N = 2048, K = 2048;

    printf("=======================================================\n");
    printf("Testing Extreme K Values (K=8 and K=128)\n");
    printf("=======================================================\n");
    printf("Matrix size: %dx%dx%d\n", M, N, K);
    printf("Instruction shape: 16x8x8 (K=8)\n");
    printf("Format: TB=64x64xK, Warp=32x32xK, Stages=2\n\n");

    // Test K=16 (baseline - known to work)
    test_kernel<typename MatMulConfig<16, 16>::Gemm>("K=16  (tb_k=16, warp_k=16)", M, N, K);

    // Test K=32 (baseline - known to work)
    test_kernel<typename MatMulConfig<32, 32>::Gemm>("K=32  (tb_k=32, warp_k=32)", M, N, K);

    // Test K=64 (baseline - known to work)
    test_kernel<typename MatMulConfig<64, 64>::Gemm>("K=64  (tb_k=64, warp_k=64)", M, N, K);

    // Test K=128 (NEW - larger value)
    test_kernel<typename MatMulConfig<128, 128>::Gemm>("K=128 (tb_k=128, warp_k=128)", M, N, K);

    printf("\n=======================================================\n");
    printf("Summary:\n");
    printf("- K=16:  Known working\n");
    printf("- K=32:  Known working (current default)\n");
    printf("- K=64:  Known working\n");
    printf("- K=128: Testing upper bound\n");
    printf("=======================================================\n");

    return 0;
}
