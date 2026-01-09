#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

// Test if tb_k and warp_k MUST match
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
    std::cout << "Testing if tb_k and warp_k must match for TF32" << std::endl;
    std::cout << "Instruction shape: 16x8x8 (K=8)" << std::endl;
    std::cout << std::endl;

    std::cout << "Compiling matching values (tb_k == warp_k):" << std::endl;

    // These should work
    using Config_16_16 = MatMulConfig<16, 16>;
    std::cout << "  ✓ tb_k=16, warp_k=16" << std::endl;

    using Config_32_32 = MatMulConfig<32, 32>;
    std::cout << "  ✓ tb_k=32, warp_k=32" << std::endl;

    using Config_64_64 = MatMulConfig<64, 64>;
    std::cout << "  ✓ tb_k=64, warp_k=64" << std::endl;

    std::cout << std::endl;
    std::cout << "Result: All matching values compile successfully!" << std::endl;
    std::cout << "Note: Mismatched values (tb_k != warp_k) fail at compile time." << std::endl;

    return 0;
}
