# Final Answer: K Dimension Constraints for TF32

## Your Question
> "You state valid K are only 16, 32, and 64. Does this mean the same limitation for both tb_k and warp_k?"

## Answer: YES - Both Must Match and Be from {16, 32, 64}

### The Complete Constraint

**For TF32 CUTLASS kernels to compile and run:**

```
tb_k == warp_k  AND  tb_k ∈ {16, 32, 64}
```

In other words:
- **tb_k and warp_k must be identical**
- **Both must be one of: 16, 32, or 64**

## Test Results

### ✅ Working Combinations (Matching Values)

| tb_k | warp_k | Status | Notes |
|------|--------|--------|-------|
| 16 | 16 | ✅ SUCCESS | Minimum valid K |
| 32 | 32 | ✅ SUCCESS | Current default |
| 64 | 64 | ✅ SUCCESS | Maximum valid K |

### ❌ Failing Combinations (Mismatched Values)

| tb_k | warp_k | Status | Error |
|------|--------|--------|-------|
| 32 | 16 | ❌ FAILS | "ThreadMap::Iterations::kColumn must be > 0" |
| 64 | 16 | ❌ FAILS | "ThreadMap::Iterations::kColumn must be > 0" |
| 64 | 32 | ❌ FAILS | "ThreadMap::Iterations::kColumn must be > 0" |

### ❌ Invalid K Values (Even When Matching)

| tb_k | warp_k | Status | Error |
|------|--------|--------|-------|
| 8 | 8 | ❌ FAILS | Too small - pipeline requires ≥2 iterations |
| 128 | 128 | ❌ FAILS | Too large - exceeds cache line constraints |

## Why Must They Match?

The compilation errors for mismatched values occur in CUTLASS's **epilogue thread mapping** logic. Specifically:

```
error: static assertion failed with
"ThreadMap::Iterations::kColumn must be > 0"
```

This error appears in the epilogue (output writing) phase, not the main computation. When `tb_k != warp_k`, CUTLASS cannot properly map the output threads to the computed results.

**Root cause**: The epilogue thread mapping expects that:
- The threadblock K dimension is tiled by warps
- Each warp processes a consistent K slice
- When `tb_k != warp_k`, the partitioning becomes uneven

## Current Implementation in autotune.py

Looking at the code:

```python
tb_k = 32  # Hardcoded

proven_warp_configs = [
    (32, 32, 32),  # warp_k always 32
    (32, 64, 32),
    (64, 32, 32),
    (64, 64, 32),
]
```

The code **always uses tb_k = warp_k = 32**, which is why this constraint wasn't immediately obvious.

## Complete Valid Configurations

For TF32, the valid (tb_k, warp_k) pairs are:

```python
valid_k_pairs = [
    (16, 16),  # Minimum
    (32, 32),  # Default (current)
    (64, 64),  # Maximum
]
```

## Summary Table

| Constraint | Requirement | Reason |
|-----------|-------------|--------|
| **Matching** | tb_k == warp_k | Epilogue thread mapping |
| **Lower bound** | K ≥ 16 | Pipeline needs ≥2 warp iterations |
| **Upper bound** | K ≤ 64 | Cache line constraint |
| **Power of 2** | K ∈ {16, 32, 64} | CUTLASS thread mapping |

## Recommendations for autotune.py

If you want to test all valid K values, modify like this:

```python
# Option 1: Keep tb_k and warp_k always matching
valid_k_values = [16, 32, 64]

for k_value in valid_k_values:
    config = KernelConfig(
        tb_m, tb_n, k_value,  # tb_k
        warp_m, warp_n, k_value,  # warp_k (must match!)
        stages,
        max_smem_kb
    )
```

**Do NOT try:**
```python
# This will NOT work - mismatched K fails!
tb_k = 64
warp_k = 32  # ❌ Compilation error
```

## Practical Impact

Since tb_k must equal warp_k, the valid K configurations are more restricted than initially thought:

- **Not**: "tb_k can be {16,32,64} and warp_k can be {16,32,64} independently"
- **Actually**: "(tb_k, warp_k) can be (16,16), (32,32), or (64,64)"

This means you have **3 valid K configurations** to test, not 9 (3×3).

## Conclusion

**Both tb_k and warp_k:**
1. Must be identical (tb_k == warp_k)
2. Must be one of: 16, 32, or 64
3. Mismatched values fail at compile time with epilogue thread mapping errors

The constraint applies **equally to both** tb_k and warp_k - they're not independent.
