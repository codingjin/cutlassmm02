#!/usr/bin/env python3
"""
CUTLASS Matrix Multiplication Auto-Tuning Script
Generates optimal kernel configurations for C = A * B
Supports: RTX 3090, RTX 4090, A100
"""

import subprocess
import itertools
import json
from dataclasses import dataclass
from typing import List, Tuple
from detect_gpu import get_gpu_config

@dataclass
class KernelConfig:
    """Kernel configuration parameters"""
    tb_m: int  # Threadblock tile M
    tb_n: int  # Threadblock tile N
    tb_k: int  # Threadblock tile K
    warp_m: int  # Warp tile M
    warp_n: int  # Warp tile N
    warp_k: int  # Warp tile K
    stages: int  # Pipeline stages
    max_smem_kb: int = 100  # Max shared memory in KB

    def is_valid(self, max_smem_kb: int = None) -> bool:
        """Check if configuration is valid"""
        # Threadblock must be divisible by warp
        if self.tb_m % self.warp_m != 0:
            return False
        if self.tb_n % self.warp_n != 0:
            return False
        if self.tb_k % self.warp_k != 0:
            return False

        # Warp must be divisible by instruction shape (16x8x8 for TF32)
        if self.warp_m % 16 != 0 or self.warp_n % 8 != 0 or self.warp_k % 8 != 0:
            return False

        # Estimate shared memory usage (bytes)
        # Shared memory ≈ (TB_M*TB_K + TB_K*TB_N) * sizeof(float) * stages * 1.2
        # 1.2× safety margin accounts for CUTLASS padding, alignment, and auxiliary buffers
        smem_per_stage = (self.tb_m * self.tb_k + self.tb_k * self.tb_n) * 4
        total_smem = smem_per_stage * self.stages * 1.2

        # Use GPU-specific shared memory limit (default 100 KB for RTX 3090/4090)
        # A100 has 164 KB/SM, but use 160 KB as practical limit
        if max_smem_kb is None:
            max_smem_kb = self.max_smem_kb
        max_smem = max_smem_kb * 1024
        if total_smem > max_smem:
            return False

        # Additional CUTLASS-specific constraints
        # Large threadblocks with many stages often fail CUTLASS validation
        tb_area = self.tb_m * self.tb_n
        if tb_area > 64 * 128 and self.stages > 3:
            return False  # 128x256, 256x128, etc. with stages 4+ fail

        return True

    def to_code(self) -> str:
        """Generate BENCHMARK_CONFIG macro call"""
        return f"BENCHMARK_CONFIG({self.tb_m}, {self.tb_n}, {self.tb_k}, " \
               f"{self.warp_m}, {self.warp_n}, {self.warp_k}, {self.stages});"


def generate_search_space(gpu_config=None) -> List[KernelConfig]:
    """Generate comprehensive search space for Ampere/Ada GPUs with proven-working warp configs

    Args:
        gpu_config: GPUConfig object from detect_gpu, or None to auto-detect
    """
    if gpu_config is None:
        gpu_config = get_gpu_config()

    # Threadblock tile sizes
    # K can be 16, 32, or 64 (must match warp_k)
    # K=32 is used as default balanced choice (see K_DIMENSION_STUDY.md)
    # Limit to smaller sizes to avoid CUTLASS validation failures
    tb_sizes = [
        (64, 64), (64, 128), (128, 64), (128, 128)
    ]
    tb_k = 32  # Valid values: 16, 32, 64 (must match warp_k)

    # Known working warp configurations for TF32
    # Format: (warp_m, warp_n, warp_k) compatible with instruction shape 16x8x8
    # NOTE: warp_k must match tb_k (all use 32 to match tb_k=32 above)
    # To test K=16 or K=64, change tb_k and all warp_k values accordingly
    proven_warp_configs = [
        (32, 32, 32),
        (32, 64, 32),
        (64, 32, 32),
        (64, 64, 32),
    ]

    # Pipeline stages
    stage_values = [2, 3, 4, 5]

    configs = []
    max_smem_kb = gpu_config.shared_mem_per_sm_kb

    for (tb_m, tb_n) in tb_sizes:
        for (warp_m, warp_n, warp_k) in proven_warp_configs:
            # Check if warp tiles divide threadblock properly
            if tb_m % warp_m != 0 or tb_n % warp_n != 0:
                continue
            if tb_k % warp_k != 0:
                continue

            for stages in stage_values:
                config = KernelConfig(
                    tb_m, tb_n, tb_k,
                    warp_m, warp_n, warp_k,
                    stages,
                    max_smem_kb
                )

                if config.is_valid(max_smem_kb):
                    configs.append(config)

    return configs


def generate_tuning_code(configs: List[KernelConfig], output_file: str):
    """Generate C++ code with all configurations"""

    template = """#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>

template<int TBM, int TBN, int TBK, int WM, int WN, int WK, int S>
struct MatMulConfig {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    static constexpr int kStages = S;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>;
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
        ElementAccumulator, OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;
};

template<typename GemmKernel>
float benchmark_matmul(int M, int N, int K, int iterations = 100) {
    using ElementA = typename GemmKernel::ElementA;
    using ElementB = typename GemmKernel::ElementB;
    using ElementC = typename GemmKernel::ElementC;

    cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<ElementB, cutlass::layout::RowMajor> B({K, N});
    cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C({M, N});

    for (int i = 0; i < M * K; i++) A.host_data()[i] = static_cast<ElementA>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) B.host_data()[i] = static_cast<ElementB>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; i++) C.host_data()[i] = static_cast<ElementC>(0);
    A.sync_device();
    B.sync_device();
    C.sync_device();

    float alpha = 1.0f, beta = 0.0f;
    typename GemmKernel::Arguments args{
        {M, N, K}, {A.device_ref()}, {B.device_ref()},
        {C.device_ref()}, {C.device_ref()}, {alpha, beta}};

    GemmKernel gemm_op;
    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return -1.0f;
    if (gemm_op.initialize(args) != cutlass::Status::kSuccess) return -1.0f;

    for (int i = 0; i < 5; ++i) gemm_op();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (gemm_op() != cutlass::Status::kSuccess) return -1.0f;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}

#define BENCHMARK_CONFIG(TBM, TBN, TBK, WM, WN, WK, STAGES) \\
    { \\
        using Config = MatMulConfig<TBM, TBN, TBK, WM, WN, WK, STAGES>; \\
        float time = benchmark_matmul<typename Config::Gemm>(M, N, K); \\
        if (time > 0) { \\
            double gflops = (2.0 * M * N * K * 1e-9) / (time * 1e-3); \\
            printf("%d,%d,%d,%d,%d,%d,%d,%.3f,%.0f\\n", \\
                   TBM, TBN, TBK, WM, WN, WK, STAGES, time, gflops); \\
            if (gflops > best_gflops) { \\
                best_gflops = gflops; \\
                best_config[0] = TBM; best_config[1] = TBN; best_config[2] = TBK; \\
                best_config[3] = WM; best_config[4] = WN; best_config[5] = WK; \\
                best_config[6] = STAGES; \\
            } \\
        } \\
    }

int main() {
    const int M = 4096, N = 4096, K = 4096;
    double best_gflops = 0.0;
    int best_config[7];

    printf("tb_m,tb_n,tb_k,warp_m,warp_n,warp_k,stages,time_ms,gflops\\n");

"""

    # Add all benchmark configs
    for config in configs:
        template += f"    {config.to_code()}\n"

    template += """
    fprintf(stderr, "\\n=== BEST CONFIGURATION ===\\n");
    fprintf(stderr, "Threadblock: %dx%dx%d\\n", best_config[0], best_config[1], best_config[2]);
    fprintf(stderr, "Warp: %dx%dx%d\\n", best_config[3], best_config[4], best_config[5]);
    fprintf(stderr, "Stages: %d\\n", best_config[6]);
    fprintf(stderr, "Performance: %.0f GFLOPS (%.1f%% of 71 TFLOPS peak)\\n",
            best_gflops, (best_gflops / 71000.0) * 100);
    return 0;
}
"""

    with open(output_file, 'w') as f:
        f.write(template)

    print(f"Generated {len(configs)} configurations in {output_file}")


if __name__ == "__main__":
    import sys

    # Detect or override GPU
    gpu_override = None
    if '--gpu' in sys.argv:
        idx = sys.argv.index('--gpu')
        if idx + 1 < len(sys.argv):
            gpu_override = sys.argv[idx + 1]

    gpu_config = get_gpu_config(gpu_override)

    print(f"Target GPU: {gpu_config.name}")
    print(f"  Compute Capability: {gpu_config.compute_cap}")
    print(f"  Architecture: {gpu_config.arch_flag}")
    print(f"  Peak TF32 TFLOPS: {gpu_config.peak_tflops_tf32}")
    print(f"  Shared Memory/SM: {gpu_config.shared_mem_per_sm_kb} KB")
    print()

    print("Generating CUTLASS auto-tuning search space...")
    configs = generate_search_space(gpu_config)
    print(f"Total configurations to test: {len(configs)}")

    generate_tuning_code(configs, "cutlass_autotune_generated.cu")
    print("\nTo compile and run:")
    print("  make autotune")
    print("  ./cutlass_autotune_generated 2>&1 | tee results.csv")
    print("\nResults will be displayed and saved to results.csv")
