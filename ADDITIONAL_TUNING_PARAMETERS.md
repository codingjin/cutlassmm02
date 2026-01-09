# Additional Tuning Parameters for CUTLASS GEMM

## Current Parameters (7)

Your current auto-tuning explores:
1. `tb_m` - Threadblock tile M dimension
2. `tb_n` - Threadblock tile N dimension
3. `tb_k` - Threadblock tile K dimension
4. `warp_m` - Warp tile M dimension
5. `warp_n` - Warp tile N dimension
6. `warp_k` - Warp tile K dimension
7. `stages` - Pipeline stages (software pipelining depth)

**Current Search Space**: 36-56 configurations (depending on GPU)

## Additional Tunable Parameters

### 1. Threadblock Swizzle Function ⭐⭐⭐⭐⭐
**Performance Impact**: HIGH (5-20% improvement possible)
**Implementation Complexity**: LOW
**Currently**: `GemmIdentityThreadblockSwizzle<>` (line 54 in your code)

#### What It Does
Controls how threadblocks are mapped to the output matrix tiles. Affects L2 cache locality and load balancing.

#### Available Options
```cpp
// Identity (current): Linear mapping, simplest
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>

// Horizontal: Better cache locality for row-major access
cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle<1>

// Custom tile count (affects cache blocking)
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>  // 2×2 tiles
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>  // 4×4 tiles
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>  // 8×8 tiles
```

#### Why It Matters
- **Cache locality**: Different swizzle patterns keep related data in L2 cache
- **Load balancing**: Can reduce tail effects when threadblocks finish at different times
- **Problem-size dependent**: Optimal swizzle varies with M, N, K sizes

#### Recommendation
**Add swizzle functor parameter** - Test values: 1, 2, 4, 8
- Small problems (M,N < 2048): swizzle=1
- Medium problems (2048 ≤ M,N < 8192): swizzle=2 or 4
- Large problems (M,N ≥ 8192): swizzle=4 or 8

### 2. Split-K Parallelism ⭐⭐⭐⭐
**Performance Impact**: MEDIUM-HIGH (10-50% for K-dominant problems)
**Implementation Complexity**: MEDIUM
**Currently**: Not used (implicit split_k_slices=1)

#### What It Does
Parallelizes the K dimension reduction across multiple threadblocks, then performs a final reduction step.

#### When Useful
- **Large K dimension**: K >> M and K >> N
- **Small M, N dimensions**: Not enough parallelism in output matrix
- **Example**: M=1024, N=1024, K=16384 → split_k=4 can give 2× speedup

#### Implementation
```cpp
// In your kernel arguments:
typename GemmKernel::Arguments args{
    {M, N, K},
    {A.device_ref()},
    {B.device_ref()},
    {C.device_ref()},
    {C.device_ref()},
    {alpha, beta},
    split_k_slices  // Add this parameter (default=1)
};
```

#### Split-K Values to Test
- `1` - No split (current behavior)
- `2` - Split K into 2 parallel reductions
- `4` - Split K into 4 parallel reductions
- `8` - Split K into 8 parallel reductions

#### Tradeoffs
- **Pro**: More parallelism, better SM utilization
- **Con**: Extra memory for partial sums, reduction overhead
- **Best for**: K >> max(M, N)

#### Recommendation
**Add split_k parameter** - But only tune for K-dominant problems:
- Case 2 in your benchmark: M=N=8192, K=16384 → Test split_k=2,4
- Case 1,3,4: M,N,K balanced → skip (split_k=1 is optimal)

### 3. Memory Alignment ⭐⭐⭐
**Performance Impact**: MEDIUM (2-10% improvement)
**Implementation Complexity**: LOW
**Currently**: Default alignment (8 for float32)

#### What It Does
Specifies alignment requirements for A, B, C matrices. Higher alignment enables vectorized memory access (128-bit or 256-bit loads).

#### Available Alignments
```cpp
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    ThreadblockSwizzle,
    kStages,
    kAlignmentA,  // Add: 1, 2, 4, 8 (elements)
    kAlignmentB,  // Add: 1, 2, 4, 8 (elements)
    // ... other parameters
>;
```

#### For Float32 (4 bytes each)
- `Alignment = 1`: 4-byte loads (32-bit)
- `Alignment = 2`: 8-byte loads (64-bit)
- `Alignment = 4`: 16-byte loads (128-bit) ✅ **Best for most cases**
- `Alignment = 8`: 32-byte loads (256-bit) ✅ **Best if guaranteed**

#### Constraints
- Input pointers must be aligned to `alignment * sizeof(ElementType)` bytes
- Leading dimensions (lda, ldb, ldc) must be multiples of alignment
- **Your case**: 4096×4096 matrices are naturally aligned to 16+ bytes

#### Recommendation
**Add alignment as tunable** - Test values: 4, 8
- Most systems support alignment=8 for float32
- Check if performance difference is measurable (likely small for large matrices)

### 4. Matrix Layouts ⭐⭐⭐⭐
**Performance Impact**: HIGH (10-40% for certain access patterns)
**Implementation Complexity**: LOW
**Currently**: All RowMajor (lines 23-25)

#### What It Does
Determines memory layout (row-major vs column-major) for each matrix.

#### Available Layouts
```cpp
// Current (all row-major)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Alternative: Column-major
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
```

#### Common Combinations
For `C = A × B`:
1. **RRR**: A row, B row, C row (current) - Common in ML frameworks
2. **RCR**: A row, B col, C row - Often fastest for square matrices
3. **CRR**: A col, B row, C row - Good for A transpose scenarios
4. **CCR**: A col, B col, C row - Less common

#### Why It Matters
- **Memory coalescing**: Layout affects global memory access patterns
- **Shared memory access**: Layout affects bank conflicts
- **Tensor Core efficiency**: Some layouts may have better instruction throughput

#### Recommendation
**Test 3-4 layout combinations**:
- Keep C as RowMajor (most common output)
- Test: RRR (current), RCR, CRR
- Each combination is a completely separate kernel compilation

### 5. Data Types and Precision ⭐⭐⭐⭐⭐
**Performance Impact**: EXTREME (2-8× speedup)
**Implementation Complexity**: MEDIUM
**Currently**: FP32 input/output with TF32 accumulation

#### Available Options

| Input Type | Output Type | Accumulator | Instruction Shape | RTX 3090 Peak | Speedup |
|------------|-------------|-------------|-------------------|---------------|---------|
| float (TF32) | float | float | 16×8×8 | 71 TFLOPS | 1× (current) |
| half (FP16) | half | float | 16×8×16 | 142 TFLOPS | 2× |
| half (FP16) | half | half | 16×8×16 | 142 TFLOPS | 2× |
| bfloat16 | bfloat16 | float | 16×8×16 | 142 TFLOPS | 2× |
| int8 | int32 | int32 | 16×8×32 | 284 TFLOPS | 4× |

#### TF32 (Current)
```cpp
using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
```

#### FP16 (2× Faster)
```cpp
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;  // FP16 input, FP32 accumulation
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
```

#### BFloat16 (2× Faster, Better Range)
```cpp
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
```

#### INT8 (4× Faster, for Inference)
```cpp
using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementAccumulator = int32_t;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
```

#### Tradeoffs
- **TF32**: Good range, decent precision (~3 decimal digits), baseline speed
- **FP16**: 2× speed, limited range (±65504), ~3 decimal digits
- **BF16**: 2× speed, same range as FP32, ~2 decimal digits, best for ML
- **INT8**: 4× speed, requires quantization, best for inference

#### Recommendation
**Create separate tuners for each precision**:
- `autotune_tf32.py` (current)
- `autotune_fp16.py` (new)
- `autotune_bf16.py` (new)
- `autotune_int8.py` (new)

Each precision mode is essentially a different tuning problem.

### 6. Epilogue Vectorization ⭐⭐
**Performance Impact**: LOW-MEDIUM (1-5%)
**Implementation Complexity**: LOW
**Currently**: 128 / cutlass::sizeof_bits<ElementC>::value (line 38)

#### What It Does
Controls how many elements the epilogue processes per instruction when writing to C.

#### Current Value
```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,  // = 4 for float32
    ElementAccumulator,
    ElementAccumulator
>;
```

For float32: `128 / 32 = 4` elements per vectorized store

#### Alternative Values
- `64 / 32 = 2` - Less aggressive vectorization
- `128 / 32 = 4` - Current (good balance)
- `256 / 32 = 8` - More aggressive (may help large outputs)

#### Recommendation
**Skip this parameter** - The default (128-bit vectorization) is optimal for most cases. Tuning gains are minimal.

### 7. Epilogue Functions ⭐⭐⭐
**Performance Impact**: MEDIUM (enables fused operations)
**Implementation Complexity**: MEDIUM-HIGH
**Currently**: LinearCombination (C = α·AB + β·C)

#### What It Does
Allows fusing additional operations into the epilogue (e.g., activation functions, bias addition).

#### Available Epilogues
```cpp
// Current: C = alpha * AB + beta * C
cutlass::epilogue::thread::LinearCombination<...>

// ReLU: C = ReLU(alpha * AB + beta * C)
cutlass::epilogue::thread::LinearCombinationRelu<...>

// GELU: C = GELU(alpha * AB + beta * C)
cutlass::epilogue::thread::LinearCombinationGELU<...>

// Bias: C = alpha * AB + bias
cutlass::epilogue::thread::LinearCombinationBias<...>

// Clamp: C = clamp(alpha * AB + beta * C, min, max)
cutlass::epilogue::thread::LinearCombinationClamp<...>
```

#### Recommendation
**Skip for now** - These are functional changes, not pure performance tuning. Useful when you need specific operations.

---

## Summary: Priority Ranking

### Tier 1: High Impact, Easy Implementation ⭐⭐⭐⭐⭐
1. **Data Types** (2-4× speedup) - Separate tuning runs for FP16/BF16/INT8
2. **Threadblock Swizzle** (5-20% speedup) - Add swizzle parameter: {1, 2, 4, 8}
3. **Matrix Layouts** (10-40% speedup) - Test RRR, RCR, CRR combinations

### Tier 2: Medium Impact, Moderate Complexity ⭐⭐⭐
4. **Split-K** (10-50% for K-dominant) - Only for Case 2 (K=16384), test {1, 2, 4}
5. **Memory Alignment** (2-10% speedup) - Test kAlignmentA/B = {4, 8}

### Tier 3: Low Impact or High Complexity ⭐
6. **Epilogue Vectorization** - Skip, current is optimal
7. **Epilogue Functions** - Skip unless you need functional changes

---

## Recommended Next Steps

### Option A: Quick Win (Swizzle + Layout)
Add 2 parameters to your current tuner:
- **Swizzle**: {1, 2, 4, 8} → ×4 search space
- **Layout**: {RRR, RCR, CRR} → ×3 search space
- **New total**: 36 configs × 4 swizzles × 3 layouts = **432 configs**

### Option B: Precision Exploration (Separate Tuners)
Create parallel tuning framework:
- `autotune_tf32.py` → 36 configs (current)
- `autotune_fp16.py` → 36 configs (2× faster)
- `autotune_bf16.py` → 36 configs (2× faster)
- Compare precision vs performance tradeoffs

### Option C: Full Expansion (All Parameters)
Ultimate search space:
- Current: 7 params → 36 configs
- + Swizzle: {1, 2, 4, 8} → ×4
- + Layout: {RRR, RCR, CRR} → ×3
- + Split-K: {1, 2, 4} → ×3
- + Alignment: {4, 8} → ×2
- **New total**: 36 × 4 × 3 × 3 × 2 = **2,592 configs**
- Runtime: ~32 hours for full benchmark (vs current ~4-5 hours)

---

## Implementation Guide

### Adding Swizzle Parameter (Easiest)

#### Step 1: Modify MatMulConfig Template
```cpp
template<
    int ThreadblockM, int ThreadblockN, int ThreadblockK,
    int WarpM, int WarpN, int WarpK, int Stages,
    int SwizzleSize  // NEW PARAMETER
>
struct MatMulConfig {
    // ... existing code ...

    using ThreadblockSwizzle =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleSize>;

    using Gemm = cutlass::gemm::device::Gemm<
        // ... existing parameters ...
        ThreadblockSwizzle,  // Use templated swizzle
        kStages
    >;
};
```

#### Step 2: Update Benchmark Macro
```cpp
#define BENCHMARK_CONFIG(TBM, TBN, TBK, WM, WN, WK, STAGES, SWIZZLE) \
    { \
        using Config = MatMulConfig<TBM, TBN, TBK, WM, WN, WK, STAGES, SWIZZLE>; \
        // ... rest of benchmark code ...
    }
```

#### Step 3: Update autotune.py
```python
def generate_search_space(gpu_config=None):
    # ... existing code ...

    swizzle_values = [1, 2, 4, 8]  # NEW

    for (tb_m, tb_n) in tb_sizes:
        for (warp_m, warp_n, warp_k) in proven_warp_configs:
            for stages in stage_values:
                for swizzle in swizzle_values:  # NEW LOOP
                    config = KernelConfig(
                        tb_m, tb_n, tb_k,
                        warp_m, warp_n, warp_k,
                        stages, swizzle,  # NEW PARAMETER
                        max_smem_kb
                    )
                    if config.is_valid(max_smem_kb):
                        configs.append(config)
```

#### Step 4: Update CSV Output
```python
printf("%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.0f\n",  # Added one %d for swizzle
       TBM, TBN, TBK, WM, WN, WK, STAGES, SWIZZLE, time, gflops);
```

### Adding Layout Parameter (More Complex)

Requires template specialization because layouts must be compile-time constants. You'd need:
```cpp
template<typename LayoutA, typename LayoutB, typename LayoutC>
struct MatMulConfigWithLayout {
    // Full struct with layout types as template parameters
};
```

Then generate separate benchmark calls for each layout combination.

---

## Expected Performance Impact

Based on CUTLASS documentation and empirical observations:

| Parameter | Typical Improvement | Problem-Dependent | Worth Tuning? |
|-----------|---------------------|-------------------|---------------|
| **Data Type** | 2-4× | No (consistent) | ⭐⭐⭐⭐⭐ YES |
| **Swizzle** | 5-20% | Yes (size-dependent) | ⭐⭐⭐⭐⭐ YES |
| **Layout** | 10-40% | Yes (access pattern) | ⭐⭐⭐⭐ YES |
| **Split-K** | 10-50% | Yes (K >> M,N) | ⭐⭐⭐ MAYBE |
| **Alignment** | 2-10% | Slightly | ⭐⭐ MAYBE |
| **Epilogue Vec** | 1-5% | No | ⭐ NO |

---

## Conclusion

Your current 7-parameter tuning is **already excellent** for TF32 GEMM. The biggest opportunities for expansion are:

1. **Precision modes** (FP16/BF16) - Different tuning problem, 2× faster
2. **Swizzle function** - Easy to add, 5-20% improvement
3. **Matrix layouts** - Moderate effort, 10-40% improvement in some cases

For a **quick win**, I recommend adding swizzle (4 values) to your current tuner:
- Increases search space from 36 to 144 configs
- Runtime increases from ~5 hours to ~20 hours for full benchmark
- Expected to find 5-15% better configurations for large matrices

Would you like me to implement the swizzle parameter addition, or explore one of the other options?
