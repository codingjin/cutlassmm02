# Investigation: K Dimension Constraints for TF32 CUTLASS

## Question
**Does tb_k (threadblock K) and warp_k really have to be 32?**

## TL;DR - Conclusion
**NO! The claim that "ThreadblockK must be 32" is INCORRECT.**

Valid K values for TF32: **16, 32, or 64** (powers of 2, K ≥ 16)

## Background

The CLAUDE.md states:
> **ThreadblockK must be 32** (not 64) - TF32 shared memory layout limitation

This investigation tested whether this constraint is accurate.

## TF32 Instruction Shape

The TF32 Tensor Core instruction shape is:
```cpp
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;  // M=16, N=8, K=8
```

This means:
- **Minimum K requirement**: warp_k % 8 == 0 (must be multiple of 8)
- **Theoretical valid K values**: 8, 16, 24, 32, 40, 48, 56, 64, ...

## Experimental Results

### Test Setup
- Matrix size: 2048×2048×2048
- Threadblock: 64×64×K
- Warp: 32×32×K
- Stages: 2
- GPU: SM86 (RTX 3090 architecture)

### Compilation Tests

| K Value | Multiple of 8? | Power of 2? | Compiles? | Runs? | Notes |
|---------|---------------|-------------|-----------|-------|-------|
| 8 | ✓ | ✓ | ❌ | - | CUTLASS error: "Number of iterations must be non-zero" |
| 16 | ✓ | ✓ | ✅ | ✅ | **Works!** |
| 24 | ✓ | ❌ | ❌ | - | CUTLASS error: divisibility/iteration constraints |
| 32 | ✓ | ✓ | ✅ | ✅ | **Works!** (current default) |
| 40 | ✓ | ❌ | ❌ | - | CUTLASS error: divisibility/iteration constraints |
| 48 | ✓ | ❌ | ❌ | - | CUTLASS error: divisibility/iteration constraints |
| 56 | ✓ | ❌ | ❌ | - | CUTLASS error: divisibility/iteration constraints |
| 64 | ✓ | ✓ | ✅ | ✅ | **Works!** |

### Pattern Discovery

**Valid K values must satisfy:**
1. K % 8 == 0 (multiple of instruction shape K=8)
2. K must be a power of 2
3. K ≥ 16 (K=8 too small for threadblock arrangement)

**Working K values: 16, 32, 64**

## Why Was K=32 Chosen as Default?

Looking at `autotune.py` (line 78-83):
```python
# Threadblock tile sizes (K=32 for TF32 compatibility)
tb_sizes = [(64, 64), (64, 128), (128, 64), (128, 128)]
tb_k = 32
```

**Likely reasons K=32 was hardcoded:**
1. **Conservative choice**: 32 is middle ground (not too small like 16, not too large like 64)
2. **Shared memory**: Larger K → more shared memory usage
   - K=16: Uses less shared memory → more configs pass validation
   - K=32: Moderate shared memory usage (current choice)
   - K=64: Uses more shared memory → fewer configs pass validation
3. **Performance sweet spot**: K=32 may have good balance for most workloads
4. **Simplicity**: Testing one K value reduces search space

## Shared Memory Impact

Estimated shared memory per stage:
```
smem ≈ (TB_M * TB_K + TB_K * TB_N) * 4 bytes
```

For threadblock 64×64:
- K=16: (64×16 + 16×64) × 4 = 8,192 bytes per stage
- K=32: (64×32 + 32×64) × 4 = 16,384 bytes per stage
- K=64: (64×64 + 64×64) × 4 = 32,768 bytes per stage

**Implication**: K=64 uses 2× the shared memory of K=32, limiting the number of pipeline stages possible.

## Recommendations

### 1. Update Documentation
Remove incorrect statements like:
- ~~"ThreadblockK must be 32 (not 64)"~~
- ~~"TF32 shared memory layout limitation"~~

Replace with:
- "ThreadblockK can be 16, 32, or 64 (powers of 2)"
- "K=32 chosen as default for balanced shared memory usage"

### 2. Expand Auto-Tuner Search Space
Consider testing K=16 and K=64 in addition to K=32:
```python
tb_k_values = [16, 32, 64]  # Test all valid K values
```

**Trade-offs:**
- K=16: Lower shared memory → more valid configs with high stages
- K=32: Current default, good balance
- K=64: Higher shared memory → fewer valid configs, but may be faster for large matrices

### 3. Performance Testing
Test performance of K=16, K=32, and K=64 on actual workloads:
- K=16 may win for memory-bound scenarios
- K=64 may win for compute-bound scenarios with large matrices
- K=32 may be best overall compromise

## Test Code

All test programs are in the repository:
- `test_k_simple.cu` - Simple compilation test (K=16, K=32)
- `test_k_powers_of_2.cu` - Confirms power-of-2 pattern
- `test_k64_only.cu` - Specific test for K=64
- `test_k_final.cu` - Comprehensive runtime test (K=16, 32, 64)

To verify:
```bash
nvcc -std=c++17 -O3 -arch=sm_86 -I/home/jin/cutlass/include \\
     --expt-relaxed-constexpr test_k_final.cu -o test_k_final
./test_k_final
```

## Conclusion

The constraint "ThreadblockK must be 32" is **historically motivated** but **technically incorrect**.

**Valid K values**: 16, 32, 64

**Why K=32 was chosen**: Practical default balancing shared memory usage and performance, not a hard technical requirement.

**Action items**:
1. Update CLAUDE.md to reflect accurate constraints
2. Consider expanding autotune search space to include K=16 and K=64
3. Profile performance across different K values for your specific workloads
