# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUTLASS-based matrix multiplication auto-tuning framework with **multi-GPU support** (RTX 3090, RTX 4090, A100). Auto-detects GPU and adapts compilation/tuning parameters. Performs exhaustive search over kernel configurations to find optimal threadblock/warp tile sizes and pipeline stages for large-scale float32 GEMM (C = A × B) using TF32 Tensor Cores.

**Supported GPUs**: RTX 3090 (SM86), RTX 4090 (SM89), A100 (SM80)
**Benchmark**: 4096×4096×4096 matmul, float32 input/output, TF32 Tensor Core execution
**Performance** (RTX 3090): 45.4 TFLOPS (63.9% of 71 TFLOPS theoretical peak)
**Auto-adapts**: Compiler flags, shared memory limits, config generation

## Build Commands

```bash
# GPU Detection
make gpu-info          # Show detected GPU and configuration

# Basic auto-tuner (12 hand-picked configs)
make                   # Build (auto-detects GPU)
make run              # Run and display results

# Extensive auto-tuner (36-56 configs depending on GPU)
python3 autotune.py                              # Generate config combinations
make autotune                                    # Compile generated code
./cutlass_autotune_generated 2>&1 | tee results.csv   # Run with live output

# Multi-size benchmark with energy measurement (4 problem sizes, powercap sweeps)
# IMPORTANT: Requires passwordless sudo for nvidia-smi (see setup below)
make run-multisize-test    # Quick test (2 configs × 5 powercaps × 4 sizes = 40 benchmarks, ~5-8 min)
make run-multisize         # Full benchmark (36 configs × 5 powercaps × 4 sizes = 720 benchmarks, ~4-5 hours)

# Data processing workflow
make norm                  # Generate normalized CSVs from summary.csv
make final                 # Generate final CSVs (combines power levels)

# Results per case folder (case1/, case2/, case3/, case4/):
#   summary.csv - Raw measurements (time, energy, GFLOPS, power)
#   norm.csv    - Normalized metrics (EDP, norm_time, norm_energy, etc.)
#   final.csv   - Pivoted data (one row per config with all power levels)

# GPU Override (for cross-compilation or testing)
make GPU=A100 autotune            # Force A100 configuration (56 configs)
make GPU="RTX 4090" multisize     # Force RTX 4090 configuration

# Correctness verification (compares CUTLASS vs cuBLAS)
make verify            # Build and run verification against cuBLAS reference

# Cleanup
make clean             # Remove binaries and generated files
```

## Architecture

### Three-Component System

1. **Basic tuner** (`cutlass_matmul_tuning.cu`): 12 hardcoded configurations, minimal compilation time
2. **Extensive tuner** (`autotune.py` → `cutlass_autotune_generated.cu`): Generates C++ with filtered search space
3. **Verification tool** (`verify_correctness.cu`): Compares CUTLASS results against cuBLAS reference to ensure correctness

### Key Constraints (RTX 3090 TF32)

**Architecture tags** (intentional mismatch):
- Compiler flag: `-arch=sm_86` (RTX 3090 hardware capability)
- CUTLASS kernel: `ArchTag = Sm80` (TF32 template works on both SM80/SM86)

**Tile dimension constraints**:
- **ThreadblockK must be 32** (not 64) - TF32 shared memory layout limitation
- Warp tiles must divide threadblock evenly
- Warp dimensions must be multiples of instruction shape (16×8×8 for TF32)

**Resource limits**:
- Shared memory: 102 KB/SM (code uses 80 KB safe limit)
- Formula: `smem ≈ (TB_M*TB_K + TB_K*TB_N) * 4 bytes * stages * 1.5`
- Large threadblocks (128×256+) with many stages (4+) hit resource limits

### Configuration Validation (`autotune.py`)

The `KernelConfig.is_valid()` method filters out configurations that would fail:
- Checks divisibility constraints (threadblock % warp, warp % instruction_shape)
- Estimates shared memory usage with 1.5× safety margin
- Rejects large threadblock areas (>128×128) with stages >3

**Proven warp configs** (guaranteed to compile):
- 32×32×32, 32×64×32, 64×32×32, 64×64×32

### Performance Characteristics

**Best configuration found**:
- Threadblock: 64×128×32
- Warp: 32×32×32
- Stages: 2
- Why: Non-square aspect ratio improves memory bandwidth, fewer stages reduce shared memory pressure

**Performance range**: 24-45 TFLOPS depending on config (basic tuner: 35.5, extensive: 45.4)

## Energy Measurement and Powercap Control

### Setup (One-Time)

The benchmark measures GPU energy consumption and tests different power caps. This requires passwordless `sudo` for `nvidia-smi`:

```bash
# Create sudoers file for passwordless nvidia-smi
sudo bash -c 'cat > /etc/sudoers.d/nvidia-smi << EOF
'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi
'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi *
EOF'

# Set correct permissions
sudo chmod 0440 /etc/sudoers.d/nvidia-smi

# Verify it works
sudo nvidia-smi -i 0 -pl 300  # Should work without password
```

### GPU-Specific Powercap Settings

The benchmark automatically detects your GPU and uses appropriate power limits:

| GPU | Powercaps (Watts) |
|-----|-------------------|
| **RTX 3090** | 100, 200, 300, 400, 450 |
| **RTX 4090** | 150, 200, 300, 400, 450 |
| **A100** | 100, 200, 250, 300, 400 |

### Measurement Protocol

**Time measurement** (3 rounds × 100 iterations):
- 10 warmup iterations (once)
- 3 measurement rounds
- CUDA events for timing
- Coefficient of variation calculated

**Energy measurement** (5 rounds × 200 iterations):
- 10 warmup iterations (once)
- 5 measurement rounds of 200 iterations each
- NVML `nvmlDeviceGetTotalEnergyConsumption()` API
- Energy reported in millijoules (mJ) with 3 decimal places
- Power calculated: `power(W) = energy(mJ) / time(ms)`

### Data Processing Workflow

```bash
# Step 1: Run benchmark (generates summary.csv)
make run-multisize-test  # or make run-multisize

# Step 2: Generate normalized metrics
make norm
# Creates case{1-4}/norm.csv with:
#   - EDP = time × energy
#   - norm_time, norm_energy (normalized to [0,1] using global min/max)
#   - norm_mul = norm_time × norm_energy
#   - norm_add = norm_time + norm_energy

# Step 3: Generate pivoted data
make final
# Creates case{1-4}/final.csv with one row per config
# Combines all 5 power levels side-by-side (41 columns total)
```

### Output Files

**summary.csv**: Raw measurement data
```
id,powercap,M,N,K,tb_m,tb_n,tb_k,warp_m,warp_n,warp_k,stages,time(ms),cv_time,gflops,energy(mj),cv_energy,power
0,100,8192,8192,8192,64,64,32,32,32,32,2,15.234,0.0023,72145,1523.456,0.0045,100
```

**norm.csv**: Normalized metrics (global normalization)
```
id,power,gflops,time,energy,EDP,norm_time,norm_energy,norm_mul,norm_add
0,100,72145,15.234,1523.456,23208.329,1,0,0,1
0,200,75432,14.567,2913.789,42445.164,0.667,0.31,0.206,0.976
```

**final.csv**: Pivoted data (one row per config)
```
id,gflops,time,energy,EDP,norm_time,norm_energy,norm_mul,norm_add,gflops,time,energy,...
0,72145,15.234,1523.456,23208.329,1,0,0,1,75432,14.567,2913.789,...
```
(8 metrics × 5 power levels = 40 columns + id = 41 columns total)

## Code Generation Flow

1. `autotune.py` generates combinations of (threadblock_size, warp_size, stages)
2. Filters via `is_valid()` to eliminate resource-violating configs
3. Writes `cutlass_autotune_generated.cu` with `BENCHMARK_CONFIG()` macro calls
4. Compilation instantiates all kernel templates (can take 1-2 minutes)
5. Runtime tests each config, outputs CSV: `tb_m,tb_n,tb_k,warp_m,warp_n,warp_k,stages,time_ms,gflops`

## Customization Points

**Matrix size**: Edit `M`, `N`, `K` constants in `.cu` files (currently 4096×4096×4096)

**Data type**: Change `ElementA/B/C` and `InstructionShape`:
- FP16: `cutlass::half_t`, instruction 16×8×16
- INT8: `int8_t`/`int32_t`, instruction 16×8×32

**Search space**: Modify `generate_search_space()` in `autotune.py`:
- `tb_sizes`: List of (M, N) threadblock dimensions
- `proven_warp_configs`: List of (warp_m, warp_n, warp_k) tuples
- `stage_values`: Pipeline stage counts to try

## Common Issues

**Compilation failures with new configs**: Warp dimensions not compatible with TF32 instruction shape (16×8×8). Stick to proven configs or ensure warp_m % 16 == 0, warp_n % 8 == 0, warp_k % 8 == 0.

**Runtime "FAILED (resource limits)"**: Config exceeds 80 KB shared memory or register limits. This is normal - extensive tuner filters these automatically via `is_valid()`.

**Low performance (<40 TFLOPS)**: Check `nvidia-smi` for thermal throttling or other GPU processes. Smaller threadblocks (64×128) with fewer stages (2-3) generally perform better than large tiles.

## Correctness Verification

The `verify_correctness.cu` program validates CUTLASS results against cuBLAS (NVIDIA's optimized GEMM library):

**What it checks**:
- Runs identical 4096×4096×4096 matmul with same inputs
- Compares ~16.7M output elements element-wise
- Reports absolute and relative errors

**Acceptance criteria** (TF32-appropriate thresholds):
- Max absolute error < 0.01
- Max relative error < 25% (allows outliers near zero)
- Average relative error < 1%
- < 0.1% of elements exceed individual thresholds (abs > 0.001 AND rel > 1%)

**Why relative error can be large**: TF32 uses 10-bit mantissa (vs 23-bit for FP32), giving ~0.1% typical precision. When dividing tiny absolute errors by small reference values near zero, relative error magnifies. Example: abs_error=0.0005, ref_value=0.003 → rel_error=16.7%. The absolute error is what matters for near-zero values.

**Typical results**: Max absolute error ~5e-4, average relative error ~0.005%, indicating excellent correctness.

## Dependencies

- CUDA 11.0+ (for TF32 support and sm_86/sm_89/sm_80 targets)
- CUTLASS headers (default: `/home/jin/cutlass/include`), override via `make CUTLASS_PATH=/path/to/cutlass`
- cuBLAS library (included with CUDA) for verification
- NVML library (nvidia-ml, included with CUDA) for energy measurement and powercap control
- Python 3 for code generation and data processing scripts:
  - `autotune.py`: Generate extensive tuning configurations
  - `generate_multisize_benchmark.py`: Generate multi-size benchmarks with energy measurement
  - `generate_norm.py`: Generate normalized CSV files from raw measurements
  - `generate_final.py`: Generate pivoted CSV files combining power levels
- `nvidia-smi` with passwordless sudo access for power cap control

## Quick Start Guide

### All-in-One Script (Easiest)

**Single command to complete the entire workflow:**

```bash
# Test mode (40 benchmarks, ~5-8 minutes)
./run_complete_benchmark.sh --test

# Full mode (720 benchmarks, ~4-5 hours)
./run_complete_benchmark.sh --full
```

This automatically handles everything from `configs` to `final.csv` with progress reporting and error checking.

### Manual Workflow Example

```bash
# 1. One-time setup: Configure passwordless nvidia-smi
sudo bash -c 'cat > /etc/sudoers.d/nvidia-smi << EOF
'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi
'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi *
EOF'
sudo chmod 0440 /etc/sudoers.d/nvidia-smi

# 2. Check GPU detection
make gpu-info

# 3. Run test benchmark (40 benchmarks, ~5-8 minutes)
#    Auto-detects GPU and uses appropriate powercap settings
make multisize-test && ./multisize_benchmark_test

# 4. Process results
make norm    # Generate normalized metrics
make final   # Generate pivoted data

# 5. View results
ls case*/summary.csv  # Raw measurements
ls case*/norm.csv     # Normalized metrics
ls case*/final.csv    # Pivoted data (one row per config)
head case1/final.csv  # Preview final data
```

### Test Mode vs Full Mode

| Mode | Configs | Powercaps | Cases | Total | Runtime | Command |
|------|---------|-----------|-------|-------|---------|---------|
| **Test** | 2 | 5 | 4 | 40 | ~5-8 min | `make run-multisize-test` |
| **Full** | 36 | 5 | 4 | 720 | ~4-5 hours | `make run-multisize` |

Both modes measure:
- Time (10 warmup + 3 rounds × 100 iterations)
- Energy (10 warmup + 5 rounds × 200 iterations)
- GFLOPS, Power, Coefficient of Variation

### Multi-GPU Systems

The benchmark automatically:
- Enables device 0 (primary GPU)
- Sets persistent mode
- Disables other GPUs to avoid interference
- All via passwordless `nvidia-smi` commands
