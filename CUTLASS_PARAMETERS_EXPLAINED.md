# CUTLASS Parameters Explained: A Complete Guide

## Table of Contents
1. [GPU Execution Hierarchy](#gpu-execution-hierarchy)
2. [The 7 Core Parameters](#the-7-core-parameters)
3. [Threadblock Swizzle Function](#threadblock-swizzle-function)
4. [Split-K Parallelism](#split-k-parallelism)
5. [Real-World Example](#real-world-example)
6. [Additional Resources](#additional-resources)

---

## GPU Execution Hierarchy

Before understanding the parameters, you need to understand how GPUs execute code:

```
┌─────────────────────────────────────────────────────────────┐
│                      GPU Device                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Streaming Multiprocessor (SM)            │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │            Threadblock (CTA)                    │  │  │
│  │  │  Contains: Multiple warps + Shared Memory      │  │  │
│  │  │  ┌───────────────────────────────────────────┐  │  │  │
│  │  │  │         Warp (32 threads)                 │  │  │  │
│  │  │  │  ┌─────────────────────────────────────┐  │  │  │  │
│  │  │  │  │  Thread (1 execution unit)          │  │  │  │  │
│  │  │  │  └─────────────────────────────────────┘  │  │  │  │
│  │  │  └───────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Levels

1. **Thread**: Single execution unit (like one CPU core)
2. **Warp**: Group of 32 threads that execute together (SIMD)
3. **Threadblock (CTA)**: Group of warps that share memory
4. **Streaming Multiprocessor (SM)**: Hardware unit that executes threadblocks
5. **GPU Device**: Contains many SMs (RTX 3090 has 82 SMs)

---

## The 7 Core Parameters

These parameters define how matrix multiplication work is divided among GPU threads.

### Matrix Multiplication: C = A × B

```
         K
    ┌─────────┐
    │         │
  M │    A    │
    │         │
    └─────────┘

         N
    ┌─────────┐
  K │    B    │
    └─────────┘

         N
    ┌─────────┐
  M │    C    │
    └─────────┘

Matrix dimensions:
- A: M × K
- B: K × N
- C: M × N

Operations: 2 × M × N × K (each C[i,j] needs K multiplies + K adds)
```

### Tiling Strategy

CUTLASS uses a **hierarchical tiling strategy** to divide the work:

```
Level 1: Grid Tiles (entire matrix C)
Level 2: Threadblock Tiles (assigned to one threadblock)
Level 3: Warp Tiles (assigned to one warp)
Level 4: Instruction Tiles (processed by Tensor Core)
```

---

### Parameter 1-3: Threadblock Tile (tb_m, tb_n, tb_k)

**What it means**: Size of the matrix tile that ONE threadblock computes.

#### Visualization

```
Full Matrix C (M × N):
┌─────────────────────────────────────────┐
│  TB    TB    TB    TB    TB    TB    TB │
│ ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐│
│ │  │  │  │  │  │  │  │  │  │  │  │  │  ││ Each box is one
│ └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘│ threadblock tile
│  TB    TB    TB    TB    TB    TB    TB │ Size: tb_m × tb_n
│ ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐│
│ │  │  │  │  │  │  │  │  │  │  │  │  │  ││
│ └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘│
└─────────────────────────────────────────┘

Example: tb_m=64, tb_n=128
Each threadblock computes a 64×128 tile of C
```

#### The K Dimension (tb_k)

For K dimension, we **iterate** because we can't load the entire K at once:

```
Computing C[64×128] = A[64×K] × B[K×128]

K dimension is split into chunks of size tb_k:

Iteration 1: C[64×128] += A[64×32] × B[32×128]  (k=0 to 31)
Iteration 2: C[64×128] += A[64×32] × B[32×128]  (k=32 to 63)
Iteration 3: C[64×128] += A[64×32] × B[32×128]  (k=64 to 95)
...
Total iterations: K / tb_k

Each iteration loads tb_k columns from A and tb_k rows from B
```

#### Key Points

✅ **Larger threadblock tiles (tb_m, tb_n)**:
- More work per threadblock → Better shared memory reuse
- More shared memory needed → May hit resource limits
- Typical values: 64×64, 64×128, 128×64, 128×128

✅ **tb_k (must be 16, 32, or 64 for TF32)**:
- Smaller → Less shared memory, more iterations
- Larger → More shared memory, fewer iterations
- **Must equal warp_k** (critical constraint!)
- Typical: tb_k=32 (balanced)

❌ **Too large**: Exceeds shared memory → kernel fails
❌ **Too small**: Poor efficiency, too many iterations

---

### Parameter 4-6: Warp Tile (warp_m, warp_n, warp_k)

**What it means**: Size of the matrix tile that ONE warp (32 threads) computes.

#### Visualization

```
One Threadblock Tile (tb_m=64, tb_n=128):
┌─────────────────────────────────────────┐
│  Warp   Warp   Warp   Warp             │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐            │
│ │32×32│ │32×32│ │32×32│ │32×32│          │ ← 4 warps in N direction
│ └────┘ └────┘ └────┘ └────┘            │
│  Warp   Warp   Warp   Warp             │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐            │
│ │32×32│ │32×32│ │32×32│ │32×32│          │ ← 4 warps in N direction
│ └────┘ └────┘ └────┘ └────┘            │
└─────────────────────────────────────────┘
  ↑
  2 warps in M direction

Example: warp_m=32, warp_n=32
Total warps per threadblock: (64/32) × (128/32) = 2 × 4 = 8 warps
Total threads per threadblock: 8 warps × 32 threads = 256 threads
```

#### Division Rules

The threadblock tile **must be evenly divisible** by warp tiles:

```
tb_m % warp_m == 0  ✅
tb_n % warp_n == 0  ✅
tb_k % warp_k == 0  ✅ AND warp_k == tb_k (for TF32)
```

**Example: tb=64×128×32, warp=32×32×32**
- M warps: 64 / 32 = 2 ✅
- N warps: 128 / 32 = 4 ✅
- K matches: 32 == 32 ✅
- Total warps: 2 × 4 = 8 warps
- Total threads: 8 × 32 = 256 threads

**Invalid example: tb=64×128×32, warp=48×48×32**
- M warps: 64 / 48 = 1.33... ❌ Not divisible!

#### Key Points

✅ **Larger warp tiles**:
- More work per warp → Better register reuse
- More registers needed → May hit limits
- Typical values: 32×32, 32×64, 64×32, 64×64

✅ **Must divide threadblock evenly**:
- Ensures all threads have work
- No wasted threads

✅ **warp_k must equal tb_k for TF32**:
- CUTLASS epilogue constraint
- See K_DIMENSION_STUDY.md for details

---

### Parameter 7: Pipeline Stages

**What it means**: Number of double-buffered tile loads in the pipeline.

#### The Problem: Memory Latency

```
Without pipelining:

┌─────────┬─────────┬─────────┬─────────┐
│  Load   │ Compute │  Load   │ Compute │  ← Compute waits for load
│  Tile 1 │ Tile 1  │ Tile 2  │ Tile 2  │
└─────────┴─────────┴─────────┴─────────┘
     ↑         ↑
   Idle     Idle
  compute   memory

Loading from global memory is SLOW (hundreds of cycles)
Tensor Cores sit IDLE waiting for data
```

#### The Solution: Software Pipelining

```
With stages=2 (double buffering):

┌─────────┬─────────┬─────────┬─────────┐
│  Load   │  Load   │  Load   │  Load   │
│  Tile 1 │  Tile 2 │  Tile 3 │  Tile 4 │  ← Memory operations
├─────────┼─────────┼─────────┼─────────┤
│         │ Compute │ Compute │ Compute │  ← Compute operations
│         │ Tile 1  │ Tile 2  │ Tile 3  │
└─────────┴─────────┴─────────┴─────────┘
           ↑
    Overlap load and compute!
    While computing tile N, load tile N+1
```

#### How It Works

```
Shared Memory Layout (stages=2):

┌─────────────────────────────────────┐
│  Stage 0 Buffer                     │
│  ┌──────────────┬──────────────┐    │
│  │  A[tb_m×tk]  │  B[tk×tb_n]  │    │
│  └──────────────┴──────────────┘    │
├─────────────────────────────────────┤
│  Stage 1 Buffer                     │
│  ┌──────────────┬──────────────┐    │
│  │  A[tb_m×tk]  │  B[tk×tb_n]  │    │
│  └──────────────┴──────────────┘    │
└─────────────────────────────────────┘

Algorithm:
1. Load tile 0 into Stage 0
2. Load tile 1 into Stage 1
3. Compute tile 0 (from Stage 0) WHILE loading tile 2 (into Stage 0)
4. Compute tile 1 (from Stage 1) WHILE loading tile 3 (into Stage 1)
5. Repeat...

Ping-pong between buffers to hide memory latency
```

#### Shared Memory Usage

```
Shared memory per stage:
smem_per_stage ≈ (tb_m × tb_k + tb_k × tb_n) × sizeof(float)

Example: tb=64×128×32, float32 (4 bytes)
smem_per_stage ≈ (64×32 + 32×128) × 4
                = (2048 + 4096) × 4
                = 24,576 bytes
                = 24 KB

Total shared memory (with safety margin):
total_smem ≈ smem_per_stage × stages × 1.2

stages=2: 24 × 2 × 1.2 = 57.6 KB ✅ (fits in 100 KB)
stages=3: 24 × 3 × 1.2 = 86.4 KB ✅ (fits in 100 KB)
stages=4: 24 × 4 × 1.2 = 115.2 KB ❌ (exceeds 100 KB limit on RTX 3090)
```

#### Key Points

✅ **More stages (2 → 3 → 4)**:
- **Pro**: Better latency hiding → Higher performance
- **Con**: More shared memory → May exceed limits

✅ **Typical values**: 2, 3, 4, 5
- RTX 3090/4090: Usually limited to 2-3 for large tiles
- A100: Can use 4-5 (has 164 KB shared memory)

✅ **Tradeoff with tile size**:
- Large tiles + many stages → Exceeds shared memory
- Small tiles + many stages → Better pipeline
- **Your job**: Find optimal balance

---

## Threadblock Swizzle Function

**What it does**: Controls the ORDER in which threadblocks are assigned to output matrix tiles.

### The Problem: L2 Cache Locality

GPUs have a large L2 cache (6 MB on RTX 3090). When computing adjacent tiles, we want to reuse data from the cache.

#### Default: Identity Swizzle

```
Matrix C (8×8 grid of threadblock tiles):

Execution order (Identity Swizzle):
┌──┬──┬──┬──┬──┬──┬──┬──┐
│0 │1 │2 │3 │4 │5 │6 │7 │  ← Row 0: Tiles executed in order
├──┼──┼──┼──┼──┼──┼──┼──┤
│8 │9 │10│11│12│13│14│15│  ← Row 1
├──┼──┼──┼──┼──┼──┼──┼──┤
│16│17│18│19│20│21│22│23│
├──┼──┼──┼──┼──┼──┼──┼──┤
│24│25│26│27│28│29│30│31│
└──┴──┴──┴──┴──┴──┴──┴──┘

Tiles are processed row-by-row, left-to-right
```

**Issue**: When tile 7 finishes, next tile is 8 (new row).
- Tile 7 uses B columns 7×tb_n to 8×tb_n
- Tile 8 uses B columns 0 to tb_n (different data!)
- Cache misses → Lower performance

#### Alternative: Horizontal Swizzle

```
Execution order (Horizontal Swizzle with block size 2):

┌──┬──┬──┬──┬──┬──┬──┬──┐
│0 │1 │4 │5 │8 │9 │12│13│  ← Process in 2×2 blocks
├──┼──┼──┼──┼──┼──┼──┼──┤
│2 │3 │6 │7 │10│11│14│15│
├──┼──┼──┼──┼──┼──┼──┼──┤
│16│17│20│21│24│25│28│29│
├──┼──┼──┼──┼──┼──┼──┼──┤
│18│19│22│23│26│27│30│31│
└──┴──┴──┴──┴──┴──┴──┴──┘

Better cache locality:
- Tile 0 → Tile 1 (same row, adjacent columns) ✅ Cache hit
- Tile 1 → Tile 2 (same block, shares B data) ✅ Cache hit
```

### Swizzle Parameter Values

```cpp
// Identity swizzle with tile size 1 (current default)
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>

// Identity swizzle with tile size 2 (process 2×2 blocks)
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>

// Identity swizzle with tile size 4 (process 4×4 blocks)
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>

// Horizontal swizzle
cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle<1>
```

### When to Use Which

| Problem Size | Matrix Shape | Recommended Swizzle | Reason |
|--------------|--------------|---------------------|--------|
| M,N < 2048 | Square | Identity<1> | Simple, low overhead |
| 2048 ≤ M,N < 8192 | Square | Identity<2> or <4> | Better cache blocking |
| M,N ≥ 8192 | Square | Identity<4> or <8> | Maximize cache reuse |
| M >> N | Tall/skinny | Horizontal<1> | Better row locality |
| N >> M | Short/wide | Identity<1> | Default works well |

### Performance Impact

**Expected improvement**: 5-20% depending on problem size and cache behavior
- **Best case**: Large matrices (8192×8192+) with good cache locality
- **Worst case**: Small matrices (<2048×2048) with little cache pressure
- **Typical**: 8-12% improvement for 4096×4096 with swizzle=4

---

## Split-K Parallelism

**What it does**: Divides the K dimension reduction across multiple threadblocks, then combines results.

### The Problem: Insufficient Parallelism

```
Small output matrix C (1024×1024) with large K (16384):

C = A[1024×16384] × B[16384×1024]

Without Split-K:
┌──────────┐
│    C     │  Only 1024×1024 / (64×128) = 128 threadblocks
│ 1024×1024│  RTX 3090 has 82 SMs → Only ~1.5 TBs per SM
└──────────┘  Many SMs sit IDLE!

K dimension is HUGE but we can't parallelize it (reduction)
```

### The Solution: Split-K

```
Split K into 4 slices (split_k_slices=4):

Slice 0: C₀ = A[1024×4096] × B[4096×1024]  (k=0 to 4095)
Slice 1: C₁ = A[1024×4096] × B[4096×1024]  (k=4096 to 8191)
Slice 2: C₂ = A[1024×4096] × B[4096×1024]  (k=8192 to 12287)
Slice 3: C₃ = A[1024×4096] × B[4096×1024]  (k=12288 to 16383)

Each slice computes partial result in parallel

Final step: C = C₀ + C₁ + C₂ + C₃ (reduction kernel)

Now: 128 TBs × 4 slices = 512 TBs → 6.2 TBs per SM ✅ Better utilization!
```

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Parallel GEMM (each slice independent)         │
└─────────────────────────────────────────────────────────┘
    A[M×K]           B[K×N]
    ┌─────┐          ┌─────┐
    │  0  │          │  0  │
    │ ... │    ×     │ ... │    →   C₀[M×N]  (partial result)
    │4095 │          │4095 │
    └─────┘          └─────┘

    ┌─────┐          ┌─────┐
    │4096 │          │4096 │
    │ ... │    ×     │ ... │    →   C₁[M×N]  (partial result)
    │8191 │          │8191 │
    └─────┘          └─────┘

    (... more slices ...)

┌─────────────────────────────────────────────────────────┐
│ Step 2: Reduction (combine partial results)            │
└─────────────────────────────────────────────────────────┘

    C₀[M×N] + C₁[M×N] + C₂[M×N] + C₃[M×N] = C[M×N]

    Lightweight kernel, much faster than GEMM
```

### Memory Overhead

```
Without Split-K:
- Output: C[M×N] in global memory

With Split-K (split_k_slices=4):
- Intermediate: C₀, C₁, C₂, C₃ (4× memory)
- Final output: C[M×N]
- Total: 5× memory vs baseline

Example: M=N=1024, float32
- Without: 1024² × 4 bytes = 4 MB
- With split_k=4: 4 × 4 MB + 4 MB = 20 MB (still small)
```

### When to Use Split-K

✅ **Use Split-K when**:
- K >> max(M, N) (K-dominant problems)
- Small output (M×N small → few threadblocks)
- Example: M=1024, N=1024, K=16384

❌ **Don't use Split-K when**:
- M, N, K balanced (e.g., M=N=K=4096)
- Large output (M×N large → already enough threadblocks)
- Memory constrained (Split-K needs extra memory)

### Choosing split_k_slices

```
General rule: Choose split_k so total threadblocks ≈ 4-8× number of SMs

RTX 3090: 82 SMs → Target 330-660 total threadblocks

Example: M=N=2048, K=16384, tb=64×128
- Without split-k: (2048/64) × (2048/128) = 32 × 16 = 512 TBs
  → Already good! split_k=1

Example: M=N=1024, K=16384, tb=64×128
- Without split-k: (1024/64) × (1024/128) = 16 × 8 = 128 TBs
  → Not enough! Try split_k=4
  → With split_k=4: 128 × 4 = 512 TBs ✅ Better!
```

**Typical values**: 1 (no split), 2, 4, 8
- split_k=1: Default, most balanced problems
- split_k=2: Slight K dominance (K ≈ 2× max(M,N))
- split_k=4: Strong K dominance (K ≈ 4× max(M,N))
- split_k=8: Extreme K dominance (K >> 8× max(M,N))

### Performance Impact

**Expected improvement**: 10-50% for K-dominant problems
- **Best case**: M=N=1024, K=16384, split_k=4 → ~40% faster
- **Worst case**: M=N=K=4096, split_k=2 → 5-10% SLOWER (overhead)
- **Your benchmark Case 2**: M=N=8192, K=16384 → Try split_k=2

---

## Real-World Example

Let's compute C[4096×4096] = A[4096×4096] × B[4096×4096] with your best config:

### Configuration
```
tb_m = 64, tb_n = 128, tb_k = 32
warp_m = 32, warp_n = 32, warp_k = 32
stages = 2
swizzle = GemmIdentityThreadblockSwizzle<1>
split_k_slices = 1
```

### Step 1: Divide Output Matrix into Threadblock Tiles

```
C[4096×4096] divided into tiles:

Number of tiles in M: 4096 / 64 = 64 tiles
Number of tiles in N: 4096 / 128 = 32 tiles
Total threadblocks: 64 × 32 = 2,048 threadblocks

Grid dimensions: gridDim = (32, 64, 1)
```

### Step 2: Each Threadblock Computes Its Tile

```
One threadblock computes C_tile[64×128]:

Number of K iterations: 4096 / 32 = 128 iterations

For each iteration i (i=0 to 127):
  1. Load A[64×32] from global memory (columns k=i×32 to k=i×32+31)
  2. Load B[32×128] from global memory (rows k=i×32 to k=i×32+31)
  3. Store in shared memory (Stage 0 or Stage 1, ping-pong)
  4. Synchronize threads (__syncthreads)
  5. Each warp computes its warp tile using Tensor Cores
  6. Accumulate result in registers

After all 128 iterations:
  7. Write C_tile[64×128] back to global memory (epilogue)
```

### Step 3: Within Each Threadblock, Divide Work Among Warps

```
Threadblock tile[64×128] divided among warps:

Warps in M dimension: 64 / 32 = 2
Warps in N dimension: 128 / 32 = 4
Total warps per threadblock: 2 × 4 = 8 warps
Total threads: 8 × 32 = 256 threads

Warp layout:
┌───────┬───────┬───────┬───────┐
│Warp 0 │Warp 1 │Warp 2 │Warp 3 │  32×32 each
├───────┼───────┼───────┼───────┤
│Warp 4 │Warp 5 │Warp 6 │Warp 7 │
└───────┴───────┴───────┴───────┘
```

### Step 4: Each Warp Computes Its Tile Using Tensor Cores

```
Warp 0 computes warp_tile[32×32]:

For each K iteration (0 to 127):
  1. Load A_fragment[32×32] from shared memory
  2. Load B_fragment[32×32] from shared memory

  3. Execute Tensor Core instructions:
     Instruction shape: 16×8×8

     Number of instructions in M: 32 / 16 = 2
     Number of instructions in N: 32 / 8 = 4
     Number of instructions in K: 32 / 8 = 4

     Total: 2 × 4 × 4 = 32 Tensor Core instructions per warp per K iteration

  4. Accumulate in warp registers

Total Tensor Core instructions per warp: 128 × 32 = 4,096 instructions
```

### Step 5: Performance Calculation

```
Total operations: 2 × M × N × K = 2 × 4096³ = 137.4 billion operations

RTX 3090 peak: 71 TFLOPS = 71 × 10¹² ops/sec

Minimum time: 137.4 × 10⁹ / (71 × 10¹²) = 1.935 ms (at 100% efficiency)

Your achieved performance: 45.4 TFLOPS
Actual time: 137.4 / 45.4 = 3.03 ms
Efficiency: 45.4 / 71 = 63.9% ✅ Excellent!
```

### Step 6: Resource Usage

```
Threads per threadblock: 256
Threadblocks per SM: Limited by shared memory and registers

Shared memory per threadblock:
  Stage 0: (64×32 + 32×128) × 4 bytes = 24,576 bytes
  Stage 1: (64×32 + 32×128) × 4 bytes = 24,576 bytes
  Total (with overhead): ~60 KB

RTX 3090 shared memory: 100 KB per SM
Max threadblocks per SM: 100 / 60 ≈ 1-2 threadblocks

Occupancy:
  2,048 total threadblocks / 82 SMs ≈ 25 threadblocks per SM
  → SMs are well-utilized ✅
```

---

## Additional Resources

### CUTLASS Official Documentation

1. **[CUTLASS GitHub](https://github.com/NVIDIA/cutlass)**
   - Main repository with code and examples

2. **[CUTLASS Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md)**
   - Official quickstart guide

3. **[GEMM API Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api.md)**
   - Detailed API reference for GEMM kernels

4. **[Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)**
   - Performance optimization guide

### NVIDIA Documentation

5. **[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)**
   - Chapter 2: Programming Model (threads, warps, blocks)
   - Chapter 5: Performance Guidelines

6. **[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)**
   - Section on Memory Optimization
   - Section on Execution Configuration

7. **[Tensor Core Programming](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)**
   - PTX documentation for Tensor Core instructions
   - WMMA (Warp Matrix Multiply-Accumulate) API

### Academic Papers

8. **"CUTLASS: Fast Linear Algebra in CUDA C++" (GTC 2018)**
   - Original CUTLASS presentation
   - [Video](https://www.nvidia.com/en-us/on-demand/session/gtcfall20-a21580/) (GTC 2020 updated version)

9. **"Dissecting the NVIDIA Volta GPU Architecture"**
   - [Blog post](https://devblogs.nvidia.com/parallelforall/inside-volta/) on Volta architecture
   - Introduces first-generation Tensor Cores

10. **"NVIDIA Ampere Architecture In-Depth"**
    - [Blog post](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) on Ampere (RTX 3090)
    - TF32 Tensor Cores explanation

### YouTube Videos

11. **"CUTLASS: Software Primitives for Dense Linear Algebra" (GTC 2020)**
    - Comprehensive tutorial on CUTLASS internals
    - Search: "GTC 2020 CUTLASS" on YouTube

12. **"CUDA Memory Model" by NVIDIA**
    - Understanding shared memory, L2 cache, global memory
    - Search: "CUDA Memory Model" on YouTube

### This Repository

13. **[TF32_AND_TENSOR_CORES_EXPLAINED.md](TF32_AND_TENSOR_CORES_EXPLAINED.md)**
    - What TF32 is and how Tensor Cores work
    - Instruction shape breakdown

14. **[K_DIMENSION_STUDY.md](K_DIMENSION_STUDY.md)**
    - Why tb_k must equal warp_k
    - Valid K values: 16, 32, 64

15. **[ADDITIONAL_TUNING_PARAMETERS.md](ADDITIONAL_TUNING_PARAMETERS.md)**
    - 7 more parameters to tune beyond the basic 7
    - Implementation guides with code examples

16. **[INSTRUCTION_SHAPES.md](INSTRUCTION_SHAPES.md)**
    - Tensor Core instruction shapes by architecture
    - TF32: 16×8×8, FP16: 16×8×16, INT8: 16×8×32

### Interactive Learning

17. **Compile and experiment with test programs in this repo**:
    ```bash
    # Basic tuner (12 configs)
    make run

    # Extensive tuner (36 configs)
    make autotune && ./cutlass_autotune_generated

    # Verify correctness
    make verify
    ```

18. **Modify parameters in cutlass_matmul_tuning.cu**:
    - Change tb_m, tb_n, tb_k values
    - Try different warp configurations
    - Experiment with stages

19. **Use NVIDIA Nsight Compute for profiling**:
    ```bash
    ncu --set full -o profile ./cutlass_matmul_tuning
    ```
    - See actual shared memory usage
    - Measure Tensor Core utilization
    - Understand bottlenecks

---

## Summary Table

| Parameter | What It Controls | Typical Values | Performance Impact | Memory Impact |
|-----------|------------------|----------------|-------------------|---------------|
| **tb_m** | Threadblock rows | 64, 128 | Medium | Shared memory |
| **tb_n** | Threadblock cols | 64, 128, 256 | Medium | Shared memory |
| **tb_k** | Threadblock K-chunk | 16, 32, 64 | Low-Medium | Shared memory |
| **warp_m** | Warp rows | 32, 64 | Medium | Registers |
| **warp_n** | Warp cols | 32, 64 | Medium | Registers |
| **warp_k** | Warp K-chunk | Must = tb_k | N/A | N/A |
| **stages** | Pipeline depth | 2, 3, 4, 5 | High | Shared memory |
| **swizzle** | Tile order | 1, 2, 4, 8 | Low-Medium | None |
| **split_k** | K parallelism | 1, 2, 4, 8 | High (K-dominant) | Global memory |

### Quick Decision Guide

**For balanced problems (M ≈ N ≈ K)**:
- Start with: tb=64×128×32, warp=32×32×32, stages=2
- swizzle=1, split_k=1
- Tune: Try tb=128×128, different warp configs, stages=3

**For K-dominant problems (K >> M, N)**:
- Start with: tb=64×128×32, warp=32×32×32, stages=2
- swizzle=1, **split_k=4**
- Tune: Adjust split_k based on (M×N) output size

**For large problems (M,N ≥ 8192)**:
- Start with: tb=128×128×32, warp=64×64×32, stages=2
- **swizzle=4**, split_k=1
- Tune: Try swizzle=8, stages=3 (if shared memory allows)

**For memory-constrained problems**:
- Use: **tb_k=16** (instead of 32), stages=2
- Smaller tiles: tb=64×64×16
- Allows more pipeline stages or larger tile dimensions

---

## Next Steps

1. **Read the materials** listed in Additional Resources
2. **Experiment** with the test programs in this repository
3. **Profile** your kernels with Nsight Compute
4. **Ask questions** - understanding these concepts takes time!

For questions specific to this project:
- Parameter constraints: See [K_DIMENSION_STUDY.md](K_DIMENSION_STUDY.md)
- Expanding search space: See [ADDITIONAL_TUNING_PARAMETERS.md](ADDITIONAL_TUNING_PARAMETERS.md)
- Quick reference: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
