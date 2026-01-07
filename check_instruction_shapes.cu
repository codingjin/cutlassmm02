#include <iostream>
#include <cutlass/gemm/gemm.h>

int main() {
    std::cout << "=== Tensor Core Instruction Shapes ===\n\n";
    
    std::cout << "Current code uses:\n";
    std::cout << "  TF32 on Ampere/Ada: GemmShape<16, 8, 8>\n";
    std::cout << "    - M=16, N=8, K=8\n";
    std::cout << "    - Used for: RTX 3090 (SM86), A100 (SM80), RTX 4090 (SM89)\n\n";
    
    std::cout << "Other instruction shapes by data type:\n";
    std::cout << "  FP16 (Volta/Turing/Ampere/Ada):\n";
    std::cout << "    - Volta (V100, SM70):    GemmShape<16, 16, 16> or <8, 8, 4>\n";
    std::cout << "    - Ampere/Ada (SM80+):    GemmShape<16, 8, 16>\n";
    std::cout << "  INT8 (Turing/Ampere/Ada):\n";
    std::cout << "    - GemmShape<16, 8, 32>\n";
    std::cout << "  FP64 (Datacenter only, A100/H100):\n";
    std::cout << "    - GemmShape<8, 8, 4>\n\n";
    
    std::cout << "Key differences:\n";
    std::cout << "  - TF32 only exists on Ampere (SM80+) and later\n";
    std::cout << "  - V100 does NOT support TF32\n";
    std::cout << "  - Instruction shape depends on BOTH arch AND data type\n";
    
    return 0;
}
