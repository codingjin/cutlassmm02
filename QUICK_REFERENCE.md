# CUTLASS Auto-Tuning Quick Reference

## Most Important Information

### ⚠️ K Dimension Constraints (CRITICAL)

**Valid K configurations**: `(tb_k, warp_k) ∈ {(16,16), (32,32), (64,64)}`

- ✅ tb_k and warp_k **must be equal**
- ✅ Both must be **16, 32, or 64**
- ❌ K=8 is too small
- ❌ K=128 is too large
- ❌ Mismatched values (e.g., tb_k=32, warp_k=16) fail

**Details**: See [K_DIMENSION_STUDY.md](K_DIMENSION_STUDY.md)

## Quick Start

### Run Complete Benchmark (Default: Full Mode)

```bash
./run_complete_benchmark.sh        # Full: 720 benchmarks (~4-5 hours)
./run_complete_benchmark.sh --test # Test: 40 benchmarks (~5-8 min)
```

**Outputs**: summary.csv → norm.csv → final.csv → final0.csv

### One-Time Setup (Required)

Configure passwordless sudo for nvidia-smi:

```bash
sudo bash -c 'cat > /etc/sudoers.d/nvidia-smi << EOF
'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi
'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi *
EOF'
sudo chmod 0440 /etc/sudoers.d/nvidia-smi
```

## Common Commands

```bash
# GPU detection
make gpu-info

# Basic tuning (12 configs)
make run

# Extensive tuning (36 configs)
make autotune && ./cutlass_autotune_generated 2>&1 | tee results.csv

# Multi-size benchmark
make run-multisize      # Full: 720 benchmarks
make run-multisize-test # Test: 40 benchmarks

# Data processing
make norm   # Generate normalized CSVs
make final  # Generate pivoted CSVs
make final0 # Generate reduced CSVs (no gflops/norm_mul)

# Correctness verification
make verify

# Clean up
make clean
```

## File Structure

```
configs                        # Input: kernel configurations
├── autotune.py               # Generate search space
├── generate_multisize_benchmark.py  # Generate benchmark code
└── run_complete_benchmark.sh # All-in-one workflow script

case1/                         # M=N=K=8192
case2/                         # M=N=8192, K=16384
case3/                         # M=N=8192, K=4096
case4/                         # M=N=16384, K=1024
├── summary.csv               # Raw measurements
├── norm.csv                  # Normalized metrics
├── final.csv                 # Pivoted (41 columns)
└── final0.csv                # Reduced (31 columns)
```

## Performance (RTX 3090)

- **Matrix size**: 4096×4096×4096 float32 TF32
- **Basic tuner**: 35.5 TFLOPS (50% efficiency)
- **Extensive tuner**: 45.4 TFLOPS (64% efficiency)
- **Best config**: TB=64×128×32, Warp=32×32×32, Stages=2

## Supported GPUs

| GPU | Configs | Shared Mem | Peak TF32 | Powercaps (W) |
|-----|---------|------------|-----------|---------------|
| RTX 3090 | 36 | 100 KB | 71 TFLOPS | 100, 200, 300, 400, 450 |
| RTX 4090 | 36 | 100 KB | 165 TFLOPS | 150, 200, 300, 400, 450 |
| A100 | 56 | 164 KB | 156 TFLOPS | 100, 200, 250, 300, 400 |

## Modifying K Dimension

To test K=16 or K=64, edit `autotune.py`:

```python
tb_k = 64  # Change from 32 to 16 or 64

proven_warp_configs = [
    (32, 32, 64),  # Change all warp_k to match tb_k
    (32, 64, 64),
    (64, 32, 64),
    (64, 64, 64),
]
```

**Remember**: tb_k and warp_k must always match!

## Output Files Explained

### summary.csv (Raw Data)
```
id,powercap,M,N,K,tb_m,tb_n,tb_k,warp_m,warp_n,warp_k,stages,
time(ms),cv_time,gflops,energy(mj),cv_energy,power
```
One row per (config, powercap) combination.

### norm.csv (Normalized)
```
id,power,gflops,time,energy,EDP,
norm_time,norm_energy,norm_mul,norm_add
```
Global normalization to [0,1] range.

### final.csv (Pivoted)
```
id,gflops,time,energy,EDP,norm_time,norm_energy,norm_mul,norm_add,
   gflops,time,energy,EDP,norm_time,norm_energy,norm_mul,norm_add,...
   (repeated for each power level)
```
One row per config, 41 columns (1 id + 8 metrics × 5 power levels).

### final0.csv (Reduced)
Same as final.csv but removes `gflops` and `norm_mul` columns.
31 columns (1 id + 6 metrics × 5 power levels).

## Troubleshooting

### Compilation Errors
```bash
# Missing CUTLASS headers
make CUTLASS_PATH=/path/to/cutlass

# Wrong GPU detected
make GPU=A100 multisize

# Check GPU
make gpu-info
```

### Runtime Errors
```bash
# "FAILED (resource limits)"
# Config exceeds shared memory - this is normal
# Autotune filters these automatically

# Low performance
nvidia-smi  # Check for thermal throttling
```

### K Dimension Errors
```bash
# "Number of iterations must be non-zero"
# K is too small (K < 16)

# "kCrosswise exceeds cache line"
# K is too large (K > 64)

# "ThreadMap::Iterations::kColumn must be > 0"
# tb_k != warp_k (they must match!)
```

## Documentation

- **[README.md](README.md)** - Complete project documentation
- **[CLAUDE.md](CLAUDE.md)** - Development guide
- **[CUTLASS_PARAMETERS_EXPLAINED.md](CUTLASS_PARAMETERS_EXPLAINED.md)** - Understanding all parameters
- **[K_DIMENSION_STUDY.md](K_DIMENSION_STUDY.md)** - K constraint analysis
- **[ADDITIONAL_TUNING_PARAMETERS.md](ADDITIONAL_TUNING_PARAMETERS.md)** - Beyond the 7 parameters
- **[TF32_AND_TENSOR_CORES_EXPLAINED.md](TF32_AND_TENSOR_CORES_EXPLAINED.md)** - TF32 fundamentals
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Full documentation index

## Need Help?

1. Check [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for complete file list
2. See [K_DIMENSION_STUDY.md](K_DIMENSION_STUDY.md) for K dimension questions
3. See [MULTI_GPU_SUPPORT.md](MULTI_GPU_SUPPORT.md) for GPU-specific issues
4. See [README.md](README.md) Troubleshooting section
