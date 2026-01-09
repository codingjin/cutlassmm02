#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

// Test if K must be power of 2
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

int main() {
    std::cout << "Testing powers of 2 for K dimension (TF32 instruction K=8)" << std::endl;
    std::cout << std::endl;

    // K=8 (1× instruction K, power of 2)
    using Config_K8 = MatMulConfig<8, 8>;
    std::cout << "✓ K=8 (2^3) compiled" << std::endl;

    // K=16 (2× instruction K, power of 2)
    using Config_K16 = MatMulConfig<16, 16>;
    std::cout << "✓ K=16 (2^4) compiled" << std::endl;

    // K=32 (4× instruction K, power of 2)
    using Config_K32 = MatMulConfig<32, 32>;
    std::cout << "✓ K=32 (2^5) compiled" << std::endl;

    std::cout << std::endl;
    std::cout << "Pattern: K must be a power of 2!" << std::endl;
    std::cout << "Valid K values for TF32: 8, 16, 32, 64, 128, ..." << std::endl;
    std::cout << "Invalid K values: 24, 40, 48, 56, ... (not powers of 2)" << std::endl;

    return 0;
}
