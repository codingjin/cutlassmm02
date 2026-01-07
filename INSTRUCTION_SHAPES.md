# Tensor Core Instruction Shapes Reference

## Summary Table

| GPU | Architecture | Compute | TF32 Shape | FP16 Shape | INT8 Shape | Notes |
|-----|--------------|---------|------------|------------|------------|-------|
| V100 | Volta | SM70 | ❌ N/A | 16×16×16 | ❌ N/A | No TF32 support |
| RTX 2080 Ti | Turing | SM75 | ❌ N/A | 16×8×8 | 16×8×32 | No TF32 support |
| A100 | Ampere | SM80 | **16×8×8** | 16×8×16 | 16×8×32 | First TF32 GPU |
| RTX 3090 | Ampere | SM86 | **16×8×8** | 16×8×16 | 16×8×32 | Current project |
| RTX 4090 | Ada | SM89 | **16×8×8** | 16×8×16 | 16×8×32 | Same as 3090 |
| H100 | Hopper | SM90 | **16×8×8** | 16×8×16 | 16×8×32 | + FP8 support |

## Current Project Configuration

```cpp
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;  // TF32
using ArchTag = cutlass::arch::Sm80;  // Works on SM80, SM86, SM89, SM90
```

**Compiler flag:** `-arch=sm_86` (RTX 3090 specific)

## Cross-GPU Portability

### ✅ Works Without Changes
- **RTX 3090** (current)
- **RTX 4090** (just change `-arch=sm_89`)
- **A100** (change `-arch=sm_80`)
- **H100** (change `-arch=sm_90`)

### ❌ Requires Code Changes
- **V100**: No TF32 → Must use FP16 with `<16,16,16>` and `Sm70`
- **RTX 2080 Ti**: No TF32 → Must use FP16 with `<16,8,8>` and `Sm75`

## Why TF32 Shape is 16×8×8

**Matrix dimensions processed per instruction:**
- **M = 16**: Output rows per instruction
- **N = 8**: Output columns per instruction
- **K = 8**: Inner dimension for reduction

**Memory layout:**
```
A tile: 16×8 (M×K) - input from matrix A
B tile: 8×8  (K×N) - input from matrix B
C tile: 16×8 (M×N) - output accumulation
```

This shape is **optimized for Ampere+ tensor core hardware** and has been kept consistent across Ampere, Ada, and Hopper architectures.

## Impact on Tuning Parameters

The instruction shape **constrains warp tile sizes**:

```python
# From autotune.py validation
if warp_m % 16 != 0 or warp_n % 8 != 0 or warp_k % 8 != 0:
    return False  # Invalid warp config
```

**Valid warp dimensions for TF32:**
- warp_m: 16, 32, 48, 64, 80, 96, ... (multiples of 16)
- warp_n: 8, 16, 24, 32, 40, 48, ... (multiples of 8)
- warp_k: 8, 16, 24, 32, 40, 48, ... (multiples of 8)

This is why your proven warp configs are:
- 32×32×32 ✓
- 32×64×32 ✓
- 64×32×32 ✓
- 64×64×32 ✓

All satisfy the instruction shape divisibility requirements!
