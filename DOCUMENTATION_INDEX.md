# CUTLASS Auto-Tuning Documentation Index

This document provides an index to all documentation files in this repository.

## Quick Start

- **[README.md](README.md)** - Main project documentation
  - Overview, prerequisites, quick start guide
  - Performance results, customization, troubleshooting
  - Multi-size benchmarking with energy measurement

- **[CLAUDE.md](CLAUDE.md)** - Instructions for Claude Code
  - Build commands, architecture overview
  - Energy measurement setup, data processing workflow
  - Quick reference for development

## Technical Deep Dives

### K Dimension Constraints (IMPORTANT)

- **[K_DIMENSION_STUDY.md](K_DIMENSION_STUDY.md)** - **Comprehensive study of tb_k and warp_k constraints**
  - ✅ Valid K values: **16, 32, or 64** (tb_k must equal warp_k)
  - ❌ Original claim "K must be 32" was incorrect
  - Detailed test results, error analysis, performance implications
  - **READ THIS** if modifying K dimensions

- **[K_COMPLETE_FINDINGS.md](K_COMPLETE_FINDINGS.md)** - Complete test results summary
  - Test results for K=8, 16, 32, 64, 128
  - Mismatched tb_k/warp_k testing
  - Quick reference table

- **[K_FINAL_ANSWER.md](K_FINAL_ANSWER.md)** - Question-answer format
  - "Does the limitation apply to both tb_k and warp_k?"
  - Direct answer with examples

### GPU Support

- **[MULTI_GPU_SUPPORT.md](MULTI_GPU_SUPPORT.md)** - Multi-GPU usage guide
  - RTX 3090, RTX 4090, A100 support
  - Auto-detection and manual override
  - GPU-specific adaptations

### Instruction Shapes and Fundamentals

- **[TF32_AND_TENSOR_CORES_EXPLAINED.md](TF32_AND_TENSOR_CORES_EXPLAINED.md)** - TF32 and Tensor Cores explained
  - What is TF32 format (19-bit floating point)
  - How Tensor Cores work (matrix multiplication hardware)
  - Instruction shape breakdown (16×8×8 for TF32)
  - Warp tile and threadblock tile hierarchy
  - Real-world examples and performance implications

- **[INSTRUCTION_SHAPES.md](INSTRUCTION_SHAPES.md)** - Tensor Core instruction reference
  - TF32, FP16, INT8 instruction shapes
  - GPU architecture comparison
  - Warp tile constraints

### Tuning Parameters

- **[CUTLASS_PARAMETERS_EXPLAINED.md](CUTLASS_PARAMETERS_EXPLAINED.md)** - **Complete guide to understanding CUTLASS parameters**
  - GPU execution hierarchy (threads → warps → threadblocks → SMs)
  - The 7 core parameters explained with visual diagrams
  - Threadblock tiles (tb_m, tb_n, tb_k) - how work is divided
  - Warp tiles (warp_m, warp_n, warp_k) - 32-thread execution units
  - Pipeline stages - software pipelining to hide memory latency
  - Threadblock swizzle function - cache locality optimization
  - Split-K parallelism - for K-dominant problems
  - Real-world example: 4096×4096 matrix multiplication
  - 19 additional learning resources (papers, videos, tutorials)
  - **START HERE** to understand the parameters

- **[ADDITIONAL_TUNING_PARAMETERS.md](ADDITIONAL_TUNING_PARAMETERS.md)** - **Beyond the current 7 parameters**
  - Analysis of additional tunable parameters in CUTLASS
  - Threadblock swizzle functions (5-20% improvement)
  - Split-K parallelism for K-dominant problems
  - Memory alignment options
  - Matrix layout combinations (RRR, RCR, CRR)
  - Data type options (FP16, BF16, INT8 for 2-4× speedup)
  - Priority ranking and implementation guide
  - **READ THIS** if expanding the search space

## Scripts and Tools

### All-in-One Workflow

- **[run_complete_benchmark.sh](run_complete_benchmark.sh)** - Complete benchmark workflow
  - Runs: configs → benchmark → summary.csv → norm.csv → final.csv → final0.csv
  - Default: Full mode (36 configs × 5 powercaps × 4 cases = 720 benchmarks)
  - Use `--test` for quick validation (40 benchmarks, ~5-8 min)

### Code Generation

- **[autotune.py](autotune.py)** - Generate extensive search space
  - Creates 36-56 configurations (GPU-dependent)
  - Filters invalid configurations
  - Generates C++ code

- **[generate_multisize_benchmark.py](generate_multisize_benchmark.py)** - Multi-size benchmark generator
  - Generates 4 different problem sizes
  - Includes energy measurement code
  - Powercap sweeping

### Data Processing

- **[generate_norm.py](generate_norm.py)** - Normalize raw measurements
  - Input: summary.csv
  - Output: norm.csv (EDP, normalized metrics)

- **[generate_final.py](generate_final.py)** - Pivot data by power level
  - Input: norm.csv
  - Output: final.csv (one row per config, 41 columns)

- **[generate_final0.py](generate_final0.py)** - Remove columns
  - Input: final.csv
  - Output: final0.csv (removes gflops and norm_mul, 31 columns)

## Build System

- **[Makefile](Makefile)** - Build system with GPU auto-detection
  - `make gpu-info` - Show detected GPU
  - `make run-multisize` - Full benchmark (720 benchmarks)
  - `make run-multisize-test` - Test benchmark (40 benchmarks)
  - `make norm` - Generate normalized CSVs
  - `make final` - Generate pivoted CSVs
  - `make final0` - Generate reduced CSVs

## Source Files

### Main Programs

- **[cutlass_matmul_tuning.cu](cutlass_matmul_tuning.cu)** - Basic auto-tuner (12 configs)
- **[verify_correctness.cu](verify_correctness.cu)** - Correctness verification vs cuBLAS
- **[multisize_benchmark.cu](multisize_benchmark.cu)** - Generated by generate_multisize_benchmark.py

### Test Programs (K Dimension Investigation)

- **[test_k_final.cu](test_k_final.cu)** - Tests K=16, 32, 64 (runtime)
- **[test_k8_runtime.cu](test_k8_runtime.cu)** - Tests K=8 (fails)
- **[test_k_extremes.cu](test_k_extremes.cu)** - Tests K=128 (fails)
- **[test_k_mismatch.cu](test_k_mismatch.cu)** - Tests mismatched tb_k/warp_k (fails)
- **[test_k_matching_only.cu](test_k_matching_only.cu)** - Tests matching values (works)
- **[test_k_powers_of_2.cu](test_k_powers_of_2.cu)** - Confirms power-of-2 pattern

## Output Files

### Per-Case Directories (case1/, case2/, case3/, case4/)

Each case represents a different problem size:
- **Case 1**: M=N=K=8192
- **Case 2**: M=N=8192, K=16384
- **Case 3**: M=N=8192, K=4096
- **Case 4**: M=N=16384, K=1024

Output files in each case:
1. **summary.csv** - Raw measurements
   - Columns: id, powercap, M, N, K, tb_m, tb_n, tb_k, warp_m, warp_n, warp_k, stages, time, cv_time, gflops, energy, cv_energy, power
   - One row per (config, powercap) pair

2. **norm.csv** - Normalized metrics
   - Columns: id, power, gflops, time, energy, EDP, norm_time, norm_energy, norm_mul, norm_add
   - Global normalization across all configs

3. **final.csv** - Pivoted by power level
   - 41 columns: id + (8 metrics × 5 power levels)
   - One row per config with all power levels side-by-side

4. **final0.csv** - Reduced columns
   - 31 columns: id + (6 metrics × 5 power levels)
   - Removes gflops and norm_mul columns

## Key Findings Summary

### K Dimension Constraints (Updated 2026-01-08)

**Valid configurations**: (tb_k, warp_k) ∈ {(16,16), (32,32), (64,64)}

| K Value | Status | Shared Memory | Use Case |
|---------|--------|---------------|----------|
| 16 | ✅ Valid | Lowest | Memory-bound, many pipeline stages |
| 32 | ✅ Valid | Moderate | **Default**, balanced performance |
| 64 | ✅ Valid | Highest | Compute-bound, large matrices |

**Invalid**:
- K=8: Too small (pipeline needs ≥2 iterations)
- K=128: Too large (exceeds cache line)
- Mismatched tb_k ≠ warp_k: Epilogue mapping fails

### Performance Results (RTX 3090)

- **Problem size**: 4096×4096×4096 float32 TF32
- **Basic tuner (12 configs)**: 35.5 TFLOPS (50.0% efficiency)
- **Extensive tuner (36 configs)**: 45.4 TFLOPS (63.9% efficiency)
- **Best config**: TB=64×128×32, Warp=32×32×32, Stages=2

### GPU-Specific Configurations

| GPU | Configs | Shared Mem | Peak TF32 | Powercaps (W) |
|-----|---------|------------|-----------|---------------|
| RTX 3090 | 36 | 100 KB | 71 TFLOPS | 100, 200, 300, 400, 450 |
| RTX 4090 | 36 | 100 KB | 165 TFLOPS | 150, 200, 300, 400, 450 |
| A100 | 56 | 164 KB | 156 TFLOPS | 100, 200, 250, 300, 400 |

## Common Tasks

### Running Full Benchmark

```bash
# Complete workflow (default: full mode)
./run_complete_benchmark.sh

# Test mode (quick validation)
./run_complete_benchmark.sh --test
```

### Modifying K Dimension

To test K=16 or K=64, edit `autotune.py`:

```python
tb_k = 64  # Change from 32 to 16 or 64

proven_warp_configs = [
    (32, 32, 64),  # Change all warp_k to match
    (32, 64, 64),
    (64, 32, 64),
    (64, 64, 64),
]
```

**Important**: tb_k and all warp_k values must match!

### Verifying Correctness

```bash
make verify
```

Compares CUTLASS vs cuBLAS with TF32-appropriate error thresholds.

## Getting Help

1. **Quick reference**: Check [CLAUDE.md](CLAUDE.md)
2. **K dimension questions**: See [K_DIMENSION_STUDY.md](K_DIMENSION_STUDY.md)
3. **Multi-GPU setup**: See [MULTI_GPU_SUPPORT.md](MULTI_GPU_SUPPORT.md)
4. **Build issues**: Check [README.md](README.md) Troubleshooting section

## Recent Updates

- **2026-01-08 (Latest)**: Comprehensive parameter guide and educational materials
  - Created CUTLASS_PARAMETERS_EXPLAINED.md (complete guide to all parameters with diagrams)
  - Created TF32_AND_TENSOR_CORES_EXPLAINED.md (comprehensive TF32/Tensor Core explanation)
  - Created ADDITIONAL_TUNING_PARAMETERS.md (7 additional tunable parameters)
  - Updated README.md, CLAUDE.md, QUICK_REFERENCE.md with new documentation
  - Enhanced documentation organization in all index files
  - Added 19 external learning resources

- **2026-01-08**: K dimension study completed, documentation updated
  - Corrected K constraints: {16, 32, 64} not just {32}
  - Added comprehensive test results
  - Updated all documentation to reflect correct constraints
  - Added final0.csv generation to workflow
