# Multi-GPU Support Guide

This project automatically adapts to **RTX 3090**, **RTX 4090**, and **A100** GPUs.

## Supported GPUs

| GPU | Compute Cap | Architecture | Peak TF32 | Shared Mem/SM | Valid Configs |
|-----|-------------|--------------|-----------|---------------|---------------|
| **RTX 3090** | 8.6 (SM86) | Ampere | 71 TFLOPS | 100 KB | 36 |
| **RTX 4090** | 8.9 (SM89) | Ada Lovelace | 165 TFLOPS | 100 KB | 36 |
| **A100** | 8.0 (SM80) | Ampere | 156 TFLOPS | 164 KB | **56** |

**Note:** A100 generates more valid configs (56 vs 36) due to larger shared memory, allowing more pipeline stages.

## Auto-Detection

The system automatically detects your GPU:

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

## Manual GPU Override

Force a specific GPU configuration:

```bash
# For A100
make GPU=A100 multisize

# For RTX 4090
make GPU="RTX 4090" multisize

# For RTX 3090
make GPU="RTX 3090" multisize
```

**Use cases:**
- Cross-compiling for a different GPU
- Testing configuration generation for other GPUs
- CI/CD pipelines

## What Gets Adapted

### 1. Compiler Architecture Flag

```bash
RTX 3090: -arch=sm_86
RTX 4090: -arch=sm_89
A100:     -arch=sm_80
```

### 2. Shared Memory Limits

**autotune.py** uses GPU-specific shared memory to filter valid configs:

```python
# RTX 3090/4090: 100 KB limit
# A100: 164 KB limit (allows more pipeline stages)
```

**Example:** A config with large threadblock (128×128) and 5 stages:
- **RTX 3090**: ❌ Invalid (exceeds 100 KB)
- **A100**: ✅ Valid (fits in 164 KB)

### 3. Peak Performance Reference

Used for performance percentage calculations:

```python
RTX 3090: 71 TFLOPS   → 45 TFLOPS = 63% of peak
RTX 4090: 165 TFLOPS  → (same kernels, ~2.3× faster)
A100:     156 TFLOPS  → (same kernels, ~2.2× faster)
```

## Code Compatibility

All three GPUs use the **same TF32 instruction shape**:

```cpp
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;  // M=16, N=8, K=8
using ArchTag = cutlass::arch::Sm80;  // Works on SM80, SM86, SM89
```

**No code changes needed** - only compiler flags differ!

## Examples

### Example 1: Auto-Detect and Run

```bash
# Detects your GPU automatically
make run-multisize-test

# Output shows detected GPU:
# Target GPU: RTX 3090
# Compiling with -arch=sm_86
```

### Example 2: Cross-Compile for A100

```bash
# On RTX 3090 machine, generate A100 configs
python3 autotune.py --gpu A100

# Generates 56 configs (vs 36 for RTX 3090)
# Output: cutlass_autotune_generated.cu

# Compile for A100
make GPU=A100 autotune
```

### Example 3: Test All GPUs

```bash
# Generate configs for each GPU
for gpu in "RTX 3090" "RTX 4090" "A100"; do
    echo "=== $gpu ==="
    python3 autotune.py --gpu "$gpu" | grep "Total configurations"
done

# Output:
# === RTX 3090 ===
# Total configurations to test: 36
# === RTX 4090 ===
# Total configurations to test: 36
# === A100 ===
# Total configurations to test: 56
```

## Performance Expectations

### Relative Performance (4096×4096×4096 TF32 GEMM)

Assuming optimal config:

| GPU | TFLOPS | Relative Speed |
|-----|--------|----------------|
| RTX 3090 | ~45 | 1.0× (baseline) |
| RTX 4090 | ~105 | **2.3×** |
| A100 | ~100 | **2.2×** |

**Note:** Actual performance depends on:
- Memory bandwidth utilization
- Optimal config for that specific GPU
- Thermal throttling
- Problem size

## Limitations

### Not Supported (Missing TF32)

- **V100** (Volta, SM70) - No TF32 Tensor Cores
- **RTX 2080 Ti** (Turing, SM75) - No TF32 support

To support these GPUs, you'd need to switch to FP16 with different instruction shapes.

## Troubleshooting

### Issue: "Unknown GPU" Warning

```
Warning: Unknown GPU 'NVIDIA Tesla V100' with compute 7.0
```

**Cause:** GPU doesn't support TF32 (compute < 8.0)

**Solution:** Use FP16 instead (requires code changes)

### Issue: Compilation Fails with Wrong Architecture

```
error: identifier "__hmma_m16n8k16_mma_f32f32" is undefined
```

**Cause:** Mismatched -arch flag and ArchTag in code

**Solution:** Ensure auto-detection is working:
```bash
make gpu-info
make clean && make multisize
```

### Issue: Different Config Count Than Expected

**RTX 3090 gives 30 configs instead of 36:**

Check if you modified shared memory limits in autotune.py:
```python
# Should be 100 for RTX 3090
max_smem_kb = gpu_config.shared_mem_per_sm_kb
```

## Implementation Details

### GPU Detection Flow

1. **detect_gpu.py** queries `nvidia-smi` for GPU name and compute capability
2. Matches against database (A100, RTX 3090, RTX 4090)
3. Returns GPUConfig with arch_flag, peak_tflops, shared_mem
4. Makefile uses this to set NVCC_FLAGS

### Adding New GPUs

To add support for H100 (SM90):

1. Edit `detect_gpu.py`:
```python
GPU_SPECS = {
    'H100': GPUConfig('H100', '9.0', 'sm_90', 500, 228),  # 500 TFLOPS, 228 KB
    # ... existing GPUs
}
```

2. That's it! No other changes needed.

The TF32 instruction shape `<16,8,8>` and `ArchTag = Sm80` work for all Ampere+ GPUs.
