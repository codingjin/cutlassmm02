# CUTLASS Matrix Multiplication Auto-Tuning

Auto-tuning framework for CUTLASS matrix multiplication (C = A × B) with **multi-GPU support** (RTX 3090, RTX 4090, A100). Automatically detects GPU and adapts compilation parameters, shared memory limits, and configuration generation.

## Overview

- **Operation**: C = A × B (matrix multiplication)
- **Benchmark Size**: 4096 × 4096 × 4096 (configurable)
- **Data Type**: float32 (using TF32 Tensor Cores)
- **Supported GPUs**:
  - RTX 3090 (Ampere, SM86) - 71 TFLOPS peak
  - RTX 4090 (Ada, SM89) - 165 TFLOPS peak
  - A100 (Ampere, SM80) - 156 TFLOPS peak
- **Auto-Detection**: Automatically detects GPU and adapts configuration
- **Achieved Performance** (RTX 3090): **45.4 TFLOPS (63.9% efficiency)**

## Prerequisites

```bash
# CUDA Toolkit 11.0+
# CUTLASS library (https://github.com/NVIDIA/cutlass)

# Install CUTLASS
git clone https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
```

## Project Structure

```
.
├── cutlass_matmul_tuning.cu         # Basic auto-tuner (12 configs)
├── autotune.py                       # Generate extensive search space
├── detect_gpu.py                     # GPU detection & configuration
├── generate_multisize_benchmark.py   # Multi-size benchmark generator
├── verify_correctness.cu             # Correctness verification vs cuBLAS
├── analyze_results.py                # Results analysis tool
├── Makefile                          # Build system with GPU auto-detection
├── CLAUDE.md                         # Claude Code instructions
├── README.md                         # This file
├── MULTI_GPU_SUPPORT.md              # Multi-GPU usage guide
└── INSTRUCTION_SHAPES.md             # Tensor Core instruction reference
```

## GPU Detection

The system automatically detects your GPU and adapts compilation:

```bash
make gpu-info
```

**Output:**
```
GPU: RTX 3090
Compute Capability: 8.6
Architecture Flag: -sm_86
Peak TF32 TFLOPS: 71
Shared Memory/SM: 100 KB
```

**GPU-Specific Adaptations:**
- **Compiler flags**: RTX 3090 uses `-arch=sm_86`, A100 uses `-arch=sm_80`, RTX 4090 uses `-arch=sm_89`
- **Shared memory limits**: RTX 3090/4090 (100 KB) → 36 configs, A100 (164 KB) → 56 configs
- **Peak performance**: Used for efficiency calculations

**Manual override:**
```bash
make GPU=A100 autotune              # Force A100 configuration
make GPU="RTX 4090" multisize       # Force RTX 4090 configuration
```

See [MULTI_GPU_SUPPORT.md](MULTI_GPU_SUPPORT.md) for detailed multi-GPU usage.

## Quick Start

### All-in-One Workflow (Recommended) 🚀

**Single command to go from `configs` to `final.csv` for all cases:**

```bash
# Full mode - ALL 36 configs (default, ~4-5 hours)
./run_complete_benchmark.sh

# Or explicitly:
./run_complete_benchmark.sh --full   # Full mode (36 configs × 5 powercaps × 4 cases = 720 benchmarks)
./run_complete_benchmark.sh --test   # Test mode (2 configs × 5 powercaps × 4 cases = 40 benchmarks, ~5-8 min)
```

**What it does automatically:**
1. ✅ Checks prerequisites (configs file, GPU, passwordless sudo)
2. ✅ Builds the appropriate benchmark
3. ✅ Runs all measurements (time + energy) across all power levels
4. ✅ Generates `summary.csv` (raw measurements)
5. ✅ Generates `norm.csv` (normalized metrics)
6. ✅ Generates `final.csv` (pivoted data - one row per config)
7. ✅ Shows results summary

**Output**: `case{1-4}/final.csv` ready for analysis!

---

### Option 1: Basic Auto-Tuning (Fast)

Tests 12 carefully selected configurations:

```bash
make
make run
```

### Option 2: Extensive Auto-Tuning (Recommended)

Generates and tests GPU-specific configurations (36 for RTX 3090/4090, 56 for A100):

```bash
python3 autotune.py       # Generate configurations (auto-detects GPU)
make autotune             # Compile
./cutlass_autotune_generated 2>&1 | tee results.csv
```

This creates:
- `results.csv` - Performance data for all configs
- Shows progress in real-time with best configuration summary

### Option 3: Multi-Size Benchmark

Test performance and energy consumption across 4 different problem sizes with power cap sweeps:
- Case 1: M=N=K=8192
- Case 2: M=N=8192, K=16384
- Case 3: M=N=8192, K=4096
- Case 4: M=N=16384, K=1024

**Setup (One-time):** Configure passwordless sudo for nvidia-smi:
```bash
sudo bash -c 'cat > /etc/sudoers.d/nvidia-smi << EOF
'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi
'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi *
EOF'
sudo chmod 0440 /etc/sudoers.d/nvidia-smi
```

**Run benchmarks:**
```bash
# Quick test (2 configs × 5 powercaps × 4 sizes = 40 benchmarks, ~5-8 min)
make run-multisize-test

# Full benchmark (36 configs × 5 powercaps × 4 sizes = 720 benchmarks, ~4-5 hours)
make run-multisize

# Process results
make norm    # Generate normalized CSV files
make final   # Generate pivoted CSV files (one row per config)
```

**GPU-Specific Power Caps:**
- RTX 3090: 100W, 200W, 300W, 400W, 450W
- RTX 4090: 150W, 200W, 300W, 400W, 450W
- A100: 100W, 200W, 250W, 300W, 400W

**Output:** Creates `case1/`, `case2/`, `case3/`, `case4/` directories with:
- `summary.csv` - Raw measurements (time, energy, GFLOPS, power, CV)
- `norm.csv` - Normalized metrics (EDP, norm_time, norm_energy, norm_mul, norm_add)
- `final.csv` - Pivoted data (one row per config with all power levels)
- `config_*_pow*.txt` - Detailed statistics for each config at each power level

**Measurement Protocol:**
- Time: 10 warmup + 3 rounds × 100 iterations (CUDA events)
- Energy: 10 warmup + 5 rounds × 200 iterations (NVML API)

### Verify Correctness

Validate CUTLASS results against cuBLAS reference implementation:

```bash
make verify
```

This compares the output of CUTLASS with cuBLAS for the same 4096×4096×4096 matrix multiplication, reporting:
- Maximum and average absolute errors
- Maximum and average relative errors (computed only for significant values)
- Number of elements exceeding thresholds

**Expected results with TF32**: Max absolute error ~5e-4, average relative error ~0.005%

### Analyze Results

```bash
python3 analyze_results.py results.csv
```

Outputs:
- Top 10 configurations
- Optimal configuration
- Statistics by pipeline stages
- Statistics by threadblock size
- Copy-paste ready code template

## Best Configuration Found

For 4096×4096 matrix multiplication on RTX 3090:

```cpp
Threadblock: 64×128×32
Warp: 32×32×32
Pipeline Stages: 2
Performance: 45.4 TFLOPS (63.9% efficiency)
Execution Time: 3.03 ms
```

**Why this works well:**
- Non-square threadblock (64×128) better utilizes memory bandwidth
- Smaller stage count (2) reduces shared memory pressure
- Fits comfortably within 80KB shared memory limit
- Good SM occupancy with multiple warps per threadblock

## Configuration Parameters

### Threadblock Tile
Controls the size of work assigned to each threadblock:
- **M, N dimensions**: 64, 128, 256
- **K dimension**: **16, 32, or 64** (must match warp_k, see K_DIMENSION_STUDY.md)
  - K=16: Lowest shared memory, most pipeline stages possible
  - K=32: Default balanced choice
  - K=64: Highest shared memory, fewer pipeline stages

Larger tiles → Better memory reuse, but more shared memory usage

### Warp Tile
Controls work per warp (32 threads):
- Must evenly divide threadblock tile
- Must be divisible by instruction shape (16×8×8 for TF32)
- **Proven configs**: 32×32×32, 32×64×32, 64×32×32, 64×64×32

### Pipeline Stages
Software pipelining depth (2-5):
- More stages → Better latency hiding
- More stages → **More shared memory usage** (main constraint)
- **Optimal**: 2-3 stages for most configurations

## Performance Results

### RTX 3090 (4096³ float32 TF32 matmul)
- **Basic tuner (12 configs)**: 35.5 TFLOPS (50.0% efficiency)
- **Extensive tuner (36 configs)**: **45.4 TFLOPS (63.9% efficiency)**
- **Improvement**: +28% performance gain
- **Best achieved**: 45.4 TFLOPS (64% of 71 TFLOPS theoretical peak)

### Expected Performance on Other GPUs
Based on relative peak TFLOPS (assuming similar efficiency):

| GPU | Peak TF32 | Expected Performance* | Relative Speed |
|-----|-----------|----------------------|----------------|
| RTX 3090 | 71 TFLOPS | 45 TFLOPS (measured) | 1.0× |
| RTX 4090 | 165 TFLOPS | ~105 TFLOPS | **2.3×** |
| A100 | 156 TFLOPS | ~100 TFLOPS | **2.2×** |

*Actual performance may vary based on memory bandwidth and optimal config for each GPU.

## Customization

### Different Matrix Sizes

Edit in source files:
```cpp
const int M = 4096;  // Your M dimension
const int N = 4096;  // Your N dimension
const int K = 4096;  // Your K dimension
```

### Different Data Types

For FP16:
```cpp
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
```

For INT8:
```cpp
using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
```

### Custom Search Space

Modify `autotune.py`:
```python
def generate_search_space() -> List[KernelConfig]:
    tb_m_values = [64, 128, 256]  # Add your values
    tb_n_values = [64, 128, 256]
    tb_k_values = [32, 64]
    # ...
```

## Build Options

```bash
# Show GPU configuration
make gpu-info

# Build all tools (basic tuner + verification)
make

# Build extensive auto-tuner (auto-detects GPU)
make autotune

# Build multi-size benchmark
make multisize              # Full (36 configs × 5 powercaps)
make multisize-test         # Test mode (2 configs × 5 powercaps)

# Run benchmarks
make run                    # Basic tuning (12 configs)
make run-autotune           # Extensive tuning (36-56 configs)
make run-multisize          # Multi-size full (720 benchmarks, ~4-5 hours)
make run-multisize-test     # Multi-size test (40 benchmarks, ~5-8 min)

# Process multi-size results
make norm                   # Generate normalized CSV files
make final                  # Generate pivoted CSV files

# Run verification
make verify

# GPU override (for cross-compilation)
make GPU=A100 autotune              # Force A100 (56 configs)
make GPU="RTX 4090" multisize       # Force RTX 4090
make GPU="RTX 3090" run-autotune    # Force RTX 3090

# Set custom CUTLASS path
make CUTLASS_PATH=/path/to/cutlass

# Combined options
make GPU=A100 CUTLASS_PATH=/path/to/cutlass autotune

# Clean build artifacts
make clean

# Help
make help
```

## Correctness Verification Details

The verification tool compares CUTLASS against cuBLAS with TF32-appropriate error thresholds:

### Why Relative Error Can Be Large

TF32 uses a **10-bit mantissa** (vs 23-bit for full FP32), providing ~3 decimal digits of precision. This means:
- **Typical relative error**: ~0.1% (2^-10)
- **Accumulated error**: Can reach several percent after 4096 accumulations

For values near zero, relative error becomes unreliable:
```
Example: absolute_error = 0.0005, reference_value = 0.003
Relative error = 0.0005 / 0.003 = 16.7%
```

The absolute error (0.0005) is tiny and acceptable, but dividing by a small number makes the relative error look large. This is why the verification tool:
- Only computes relative error for |value| > 0.001
- Accepts max relative error up to 25% (handles edge cases)
- Requires average relative error < 1% (ensures overall quality)

### Verification Pass Criteria

- **Max absolute error** < 0.01 (good absolute precision)
- **Max relative error** < 25% (tolerates outliers near zero)
- **Avg relative error** < 1% (overall quality check)
- **Bad elements** < 0.1% (few elements with abs > 0.001 AND rel > 1%)

### Typical Results

```
Total elements:         16777216
Significant values:     16774723 (|val| > 1e-3)
Max absolute error:     5.289e-04
Max relative error:     18.27% (1-2 outliers near zero)
Avg absolute error:     4.618e-05
Avg relative error:     0.005%
Errors > threshold:     0
Status:                 ✓ VERIFICATION PASSED
```

## Important Notes

### Architecture Tags: Auto-Adapted

The system automatically sets the correct architecture flag based on detected GPU:
- **RTX 3090**: `-arch=sm_86` (compiler) + `ArchTag = Sm80` (CUTLASS template)
- **RTX 4090**: `-arch=sm_89` (compiler) + `ArchTag = Sm80` (CUTLASS template)
- **A100**: `-arch=sm_80` (compiler) + `ArchTag = Sm80` (CUTLASS template)

**Why ArchTag is always Sm80:** The TF32 CUTLASS kernel templates are the same for SM80/86/89. They all use the same TF32 instruction shape `<16,8,8>` and work across all three architectures.

### Resource Limits

Configurations fail with "resource limits" when they exceed GPU-specific constraints:

**Shared memory per SM:**
- RTX 3090/4090: 100 KB (practical limit ~80 KB with 1.2× safety margin)
- A100: 164 KB (practical limit ~136 KB with 1.2× safety margin)

This is why A100 generates 56 configs vs 36 for RTX 3090/4090—its larger shared memory allows more pipeline stages.

**Other limits:**
- **Register pressure**: Too many registers per thread
- **Occupancy**: Can't launch enough warps

**Common failures:**
- Large threadblocks (256×256) with many stages (4+)
- Mismatched tb_k and warp_k (they must be equal)
- Invalid K values (K must be 16, 32, or 64)

The auto-tuner filters these automatically based on detected GPU.

## Troubleshooting

### Compilation Errors

**Error: CUTLASS headers not found**
```bash
# Set CUTLASS_PATH
export CUTLASS_PATH=/path/to/cutlass
make CUTLASS_PATH=$CUTLASS_PATH
```

**Error: arch=sm_XX not supported**
```bash
# Check CUDA version (need 11.0+)
nvcc --version

# Verify GPU detection
make gpu-info

# Manual override if needed
make GPU=A100 autotune

# Architecture mapping:
# sm_86 = RTX 3090 (Ampere consumer)
# sm_89 = RTX 4090 (Ada Lovelace)
# sm_80 = A100 (Ampere datacenter)
# sm_90 = H100 (Hopper, not yet supported)
```

**Error: "Unknown GPU" warning**
```
Warning: Unknown GPU 'NVIDIA Tesla V100' with compute 7.0
```
Your GPU doesn't support TF32 (requires compute capability 8.0+). This framework only supports:
- RTX 3090 (SM86)
- RTX 4090 (SM89)
- A100 (SM80)

For older GPUs, you'd need to modify the code to use FP16 instead of TF32.

### Runtime Errors

**"FAILED (resource limits)"**
- Configuration exceeds shared memory or register limits
- This is normal and expected for large configurations
- The extensive auto-tuner filters these out automatically

**"FAILED (init error)"**
- Kernel initialization error (rare)
- Usually indicates invalid parameter combination

**Low performance**
- Check GPU is not thermal throttling: `nvidia-smi`
- Ensure no other processes using GPU
- Verify TF32 mode enabled (default in CUDA 11+)
- Try smaller threadblocks with fewer stages

## Understanding Output

```
TB:128x128x32 | W:64x64x32 | S:3 => 2.156 ms | 63847.23 GFLOPS
```

- **TB**: Threadblock tile (M×N×K)
- **W**: Warp tile (M×N×K)
- **S**: Pipeline stages
- **Time**: Average execution time
- **GFLOPS**: Billion floating-point operations per second

## Additional Documentation

### Getting Started
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page quick reference
  - Most important constraints (K dimension)
  - Common commands
  - File structure overview
  - Troubleshooting guide

- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation index
  - All documentation files organized by category
  - Common tasks and workflows
  - Recent updates log

### Technical Deep Dives

- **[CUTLASS_PARAMETERS_EXPLAINED.md](CUTLASS_PARAMETERS_EXPLAINED.md)** - **Complete guide to CUTLASS parameters**
  - GPU execution hierarchy (threads → warps → threadblocks)
  - The 7 core parameters with visual diagrams
  - Threadblock tiles, warp tiles, pipeline stages
  - Threadblock swizzle and Split-K parallelism
  - Real-world 4096×4096 example walkthrough
  - 19 additional learning resources
  - **START HERE to understand the parameters**

- **[TF32_AND_TENSOR_CORES_EXPLAINED.md](TF32_AND_TENSOR_CORES_EXPLAINED.md)** - TF32 and Tensor Cores explained
  - What is TF32 format (19-bit floating point)
  - How Tensor Cores work (matrix multiplication hardware)
  - Instruction shape breakdown (16×8×8 for TF32)
  - Why K dimension constraints exist

- **[K_DIMENSION_STUDY.md](K_DIMENSION_STUDY.md)** - K dimension constraint study
  - Valid K values: 16, 32, or 64 (tb_k must equal warp_k)
  - Comprehensive test results
  - Error analysis and explanations
  - Performance implications

- **[ADDITIONAL_TUNING_PARAMETERS.md](ADDITIONAL_TUNING_PARAMETERS.md)** - Beyond the current 7 parameters
  - Analysis of 7 additional tunable parameters
  - Threadblock swizzle (5-20% improvement potential)
  - Split-K parallelism for K-dominant problems
  - Matrix layouts, memory alignment, data types (FP16/BF16/INT8)
  - Priority ranking and implementation guide

### GPU and Hardware

- **[MULTI_GPU_SUPPORT.md](MULTI_GPU_SUPPORT.md)** - Comprehensive multi-GPU usage guide
  - GPU detection and override
  - Cross-compilation examples
  - Performance expectations
  - Adding new GPU support

- **[INSTRUCTION_SHAPES.md](INSTRUCTION_SHAPES.md)** - Tensor Core instruction reference
  - Instruction shapes by GPU architecture
  - Data type compatibility
  - Warp tile constraints

### Development

- **[CLAUDE.md](CLAUDE.md)** - Project overview and build commands for Claude Code

## References

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUTLASS Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)
- [Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [Ada Lovelace Architecture](https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/)
- [TF32 Format](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)

## License

This code uses CUTLASS which is licensed under the BSD 3-Clause License.
