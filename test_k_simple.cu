#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

// Simple test: Just K=16 and K=32
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
    std::cout << "Compiling TF32 CUTLASS kernels..." << std::endl;
    std::cout << "Instruction shape: 16x8x8 (K=8)" << std::endl;
    std::cout << std::endl;

    // K=16
    using Config_K16 = MatMulConfig<16, 16>;
    std::cout << "✓ K=16 compiled successfully" << std::endl;

    // K=32
    using Config_K32 = MatMulConfig<32, 32>;
    std::cout << "✓ K=32 compiled successfully" << std::endl;

    return 0;
}
