# TF32 and Tensor Cores Explained

## What is TF32?

**TF32 (TensorFloat-32)** is a special floating-point format introduced by NVIDIA with the Ampere GPU architecture (2020).

### TF32 Format Breakdown

```
┌─────────────────────────────────────────┐
│     TF32: 19 bits total                 │
├──────┬──────────────────┬───────────────┤
│ Sign │ Exponent         │ Mantissa      │
│ 1    │ 8 bits           │ 10 bits       │
└──────┴──────────────────┴───────────────┘

Compare to:

┌─────────────────────────────────────────┐
│    FP32 (standard float): 32 bits       │
├──────┬──────────────────┬───────────────┤
│ Sign │ Exponent         │ Mantissa      │
│ 1    │ 8 bits           │ 23 bits       │
└──────┴──────────────────┴───────────────┘

┌─────────────────────────────────────────┐
│    FP16 (half float): 16 bits           │
├──────┬──────────────────┬───────────────┤
│ Sign │ Exponent         │ Mantissa      │
│ 1    │ 5 bits           │ 10 bits       │
└──────┴──────────────────┴───────────────┘
```

### Key Characteristics

| Format | Bits | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| **TF32** | 19 | 8 bits | 10 bits | Same as FP32 | ~3 decimal digits |
| FP32 | 32 | 8 bits | 23 bits | ±3.4×10³⁸ | ~7 decimal digits |
| FP16 | 16 | 5 bits | 10 bits | ±6.5×10⁴ | ~3 decimal digits |

**The Magic of TF32**:
- **Same range as FP32** (8-bit exponent)
- **Similar precision to FP16** (10-bit mantissa)
- **Best of both worlds**: Wide range + reasonable precision + fast computation

### Why TF32?

**Problem**: AI/ML workloads need:
1. Fast computation (like FP16)
2. Wide numeric range (like FP32)
3. Good numerical stability

**Solution**: TF32 provides:
- **8× faster** than FP32 on Tensor Cores
- **Same dynamic range** as FP32 (no overflow issues)
- **Automatic**: Works transparently with FP32 code
- **Good enough precision** for most ML/scientific workloads

### How TF32 Works in Practice

```python
# Your Python/C++ code uses regular float32:
float A[M*K];  # FP32 data in memory (32 bits)
float B[K*N];  # FP32 data in memory (32 bits)
float C[M*N];  # FP32 results in memory (32 bits)

# What happens inside GPU Tensor Cores:
# 1. Load FP32 data from memory
# 2. Convert to TF32 format (automatic, in hardware)
# 3. Compute using TF32 Tensor Core instructions
# 4. Accumulate in FP32
# 5. Write FP32 results back to memory

# Result: You get FP32 convenience + TF32 speed!
```

## What are Tensor Cores?

**Tensor Cores** are specialized hardware units in NVIDIA GPUs designed for **matrix multiplication**.

### Evolution

| GPU Generation | Architecture | Tensor Cores | Supported Formats |
|----------------|--------------|--------------|-------------------|
| Volta (V100) | 2017 | 1st Gen | FP16 only |
| Turing (RTX 20) | 2018 | 2nd Gen | FP16, INT8, INT4 |
| **Ampere (A100, RTX 30)** | **2020** | **3rd Gen** | **TF32**, FP16, BF16, FP64, INT8 |
| Ada (RTX 40) | 2022 | 4th Gen | TF32, FP16, FP8, INT8 |
| Hopper (H100) | 2022 | 4th Gen+ | TF32, FP16, FP8, INT8 |

### Regular CUDA Core vs Tensor Core

**Regular CUDA Core**:
```
One operation per clock:
A * B + C = D  (one FMA operation)
```

**Tensor Core**:
```
One operation per clock:
[4×4 matrix] × [4×4 matrix] + [4×4 matrix] = [4×4 matrix]
That's 64 FMA operations in one clock cycle!

Speed up: ~8× faster than CUDA cores for matrix multiplication
```

### Visual Comparison

```
CUDA Core (scalar):
┌───┐   ┌───┐       ┌───┐
│ A │ × │ B │ + C = │ D │
└───┘   └───┘       └───┘
1 FMA operation per cycle

Tensor Core (matrix):
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│ A B │ × │ E F │ + │ I J │ = │ M N │
│ C D │   │ G H │   │ K L │   │ O P │
└─────┘   └─────┘   └─────┘   └─────┘
Multiple FMA operations per cycle
```

## What is "Instruction Shape"?

The **instruction shape** defines the dimensions of matrices that a Tensor Core can process **in a single instruction**.

### TF32 Instruction Shape: 16×8×8

```cpp
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
//                                                 ↑   ↑  ↑
//                                                 M   N  K
```

This means **one Tensor Core instruction** performs:

```
┌──────────────┐   ┌─────────┐   ┌──────────────┐
│              │   │         │   │              │
│  A matrix    │ × │ B matrix│ = │  C matrix    │
│  16 × 8      │   │ 8 × 8   │   │  16 × 8      │
│              │   │         │   │              │
└──────────────┘   └─────────┘   └──────────────┘

One instruction computes:
C[16×8] = A[16×8] × B[8×8] + C[16×8]
```

### Breaking Down the Instruction Shape

**M=16**: Output has 16 rows
**N=8**: Output has 8 columns
**K=8**: Inner dimension for dot product

**What happens inside**:
```
For each element C[i,j]:
  C[i,j] = sum(A[i,k] * B[k,j])  for k=0 to 7

Total operations: 16 × 8 × (8 multiplies + 8 adds) = 2,048 operations
In one instruction cycle!
```

### Why These Specific Numbers?

The instruction shape is designed to:
1. **Fit in hardware**: Tensor Core has physical layout for 16×8×8
2. **Balance workload**: Good ratio of M:N:K for memory bandwidth
3. **Enable tiling**: Can be composed into larger matrix multiplications

## Warp Tile and Threadblock Tile

Now we can understand the hierarchy in CUTLASS:

### Level 1: Instruction (Hardware)
```
Instruction Shape: 16×8×8
One Tensor Core instruction
Fixed by hardware
```

### Level 2: Warp Tile (32 threads)
```
Warp Shape: 32×32×32 (for example)
How many instructions per warp?
  M: 32/16 = 2 instructions
  N: 32/8  = 4 instructions
  K: 32/8  = 4 iterations
```

**A warp tile must**:
- Be divisible by instruction shape
- Coordinate 32 threads (one warp)
- Fit in registers

### Level 3: Threadblock Tile (multiple warps)
```
Threadblock Shape: 64×128×32 (for example)
How many warps?
  M: 64/32  = 2 warp tiles in M
  N: 128/32 = 4 warp tiles in N
  Total: 2×4 = 8 warps per threadblock
```

**A threadblock tile must**:
- Be divisible by warp shape
- Coordinate multiple warps
- Fit in shared memory

### Visual Hierarchy

```
┌─────────────────────────────────────────┐
│     Threadblock Tile: 64×128×32         │
│                                          │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ Warp Tile    │  │ Warp Tile    │    │
│  │ 32×32×32     │  │ 32×32×32     │    │
│  │              │  │              │    │
│  │ ┌──┐ ┌──┐   │  │ ┌──┐ ┌──┐   │    │
│  │ │16│ │16│   │  │ │16│ │16│   │    │
│  │ │×8│ │×8│   │  │ │×8│ │×8│   │    │
│  │ └──┘ └──┘   │  │ └──┘ └──┘   │    │
│  │ Instruction  │  │ Instruction  │    │
│  │ Shape        │  │ Shape        │    │
│  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────┘
```

## The K Dimension Constraint Explained

Now you can understand **why K must be {16, 32, 64}**:

### Instruction K = 8

Each Tensor Core instruction processes K=8 elements.

### Warp K Constraint

```
Warp K must be a multiple of instruction K=8:
  K=8:  8/8  = 1 iteration (too few!)
  K=16: 16/8 = 2 iterations ✅
  K=32: 32/8 = 4 iterations ✅
  K=64: 64/8 = 8 iterations ✅
  K=128: 128/8 = 16 iterations (too many for cache line!)
```

**Why K=8 fails**: CUTLASS pipeline structure requires at least 2 iterations in the K dimension.

**Why K=128 fails**: Exceeds shared memory cache line size for the TF32 tensor op layout.

**Why K must be power of 2**: CUTLASS thread mapping logic requires power-of-2 for efficient memory access patterns.

## Real Example: Matrix Multiplication

Let's compute `C[4096×4096] = A[4096×4096] × B[4096×4096]`:

### Step 1: Choose Configuration
```
Threadblock: 64×128×32
Warp:        32×32×32
Instruction: 16×8×8 (fixed by hardware)
Stages:      2
```

### Step 2: Calculate Grid Dimensions
```
Threadblocks needed:
  M direction: 4096 / 64  = 64 threadblocks
  N direction: 4096 / 128 = 32 threadblocks
  Grid: 64 × 32 = 2,048 threadblocks
```

### Step 3: Within Each Threadblock
```
Warps per threadblock:
  M: 64/32  = 2
  N: 128/32 = 4
  Total: 2×4 = 8 warps

Each warp executes:
  M: 32/16 = 2 instruction rows
  N: 32/8  = 4 instruction columns
  K: 32/8  = 4 instruction iterations
  Total: 2×4×4 = 32 Tensor Core instructions per warp
```

### Step 4: K Iterations
```
Outer K loop: 4096 / 32 = 128 iterations
Each iteration:
  - Load A[64×32] tile from global memory to shared memory
  - Load B[32×128] tile from global memory to shared memory
  - Compute C[64×128] += A[64×32] × B[32×128] using Tensor Cores
```

## Performance Implications

### Why TF32 is Fast

**RTX 3090 Example**:
- **FP32 performance**: 35 TFLOPS (CUDA cores)
- **TF32 performance**: 71 TFLOPS (Tensor Cores)
- **Speedup**: 2× faster!

**This project achieves**: 45.4 TFLOPS = 64% of peak

### What Affects Performance

1. **Tile sizes**: Larger tiles → better memory reuse → higher performance
2. **K dimension**: K=32 is balanced, K=16 uses less memory, K=64 uses more
3. **Pipeline stages**: More stages → better latency hiding → higher performance (but more memory)
4. **Warp arrangement**: Good warp tile → better SM utilization

## Summary

### TF32
- **What**: 19-bit floating-point format (8-bit exp, 10-bit mantissa)
- **Why**: Fast like FP16, range like FP32
- **Where**: Tensor Cores in Ampere+ GPUs
- **Speed**: ~8× faster than FP32 on CUDA cores

### Tensor Cores
- **What**: Specialized hardware for matrix multiplication
- **How**: Process small matrix tiles in one instruction
- **Speed**: Many operations per clock cycle

### Instruction Shape (16×8×8)
- **What**: Size of matrices one Tensor Core instruction can process
- **Fixed by**: GPU hardware architecture
- **Defines**: Minimum tile sizes for warp and threadblock

### K Dimension
- **Instruction K**: 8 (fixed by hardware)
- **Warp/Threadblock K**: Must be multiple of 8
- **Valid values**: 16, 32, 64 (due to pipeline and cache constraints)
- **Must match**: tb_k must equal warp_k

## References

- [NVIDIA TF32 Blog](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
- [Ampere Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- This project: See K_DIMENSION_STUDY.md for constraint details

## Visual Summary

```
┌──────────────────────────────────────────────────────────┐
│  Matrix Multiplication: C = A × B                        │
│  Size: 4096 × 4096 × 4096                                │
└──────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  Grid of Threadblocks         │
        │  64 × 32 = 2,048 blocks       │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  One Threadblock              │
        │  Shape: 64 × 128 × 32         │
        │  Warps: 8                     │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  One Warp (32 threads)        │
        │  Shape: 32 × 32 × 32          │
        │  Instructions: 32             │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  One Tensor Core Instruction  │
        │  Shape: 16 × 8 × 8 (TF32)     │
        │  Hardware: Fixed              │
        └───────────────────────────────┘
```

Each level must be divisible by the level below it!
