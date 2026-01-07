#!/usr/bin/env python3
"""Generate benchmark code for multiple problem sizes with statistical analysis"""

import math

def read_configs(filename='configs'):
    """Read configurations from file"""
    configs = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) == 7:
                configs.append({
                    'tb_m': int(parts[0]),
                    'tb_n': int(parts[1]),
                    'tb_k': int(parts[2]),
                    'warp_m': int(parts[3]),
                    'warp_n': int(parts[4]),
                    'warp_k': int(parts[5]),
                    'stages': int(parts[6])
                })
    return configs

def generate_benchmark_code(configs, problem_sizes, output_file, test_mode=False):
    """Generate C++ benchmark code with statistical analysis

    Args:
        configs: List of kernel configurations
        problem_sizes: Dictionary of problem sizes
        output_file: Output filename
        test_mode: If True, only generate first 2 configs per case for quick testing
    """

    template_header = """#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <nvml.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>

// GPU type enumeration
enum class GPUType {
    RTX_3090,
    RTX_4090,
    A100,
    UNKNOWN
};

// GPU configuration structure
struct GPUInfo {
    GPUType type;
    std::string name;
    std::vector<unsigned int> powercaps;  // in milliwatts
};

// Detect GPU type using NVML
GPUType detectGPU(nvmlDevice_t device) {
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    if (nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE) != NVML_SUCCESS) {
        return GPUType::UNKNOWN;
    }

    std::string gpuName(name);
    if (gpuName.find("RTX 3090") != std::string::npos) {
        return GPUType::RTX_3090;
    } else if (gpuName.find("RTX 4090") != std::string::npos) {
        return GPUType::RTX_4090;
    } else if (gpuName.find("A100") != std::string::npos) {
        return GPUType::A100;
    }
    return GPUType::UNKNOWN;
}

// Get GPU information including powercap settings
GPUInfo getGPUInfo(nvmlDevice_t device) {
    GPUInfo info;
    info.type = detectGPU(device);

    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    info.name = std::string(name);

    // Define powercap settings for each GPU type (in milliwatts)
    switch (info.type) {
        case GPUType::RTX_3090:
            info.powercaps = {100000, 200000, 300000, 400000, 450000};
            break;
        case GPUType::RTX_4090:
            info.powercaps = {150000, 200000, 300000, 400000, 450000};
            break;
        case GPUType::A100:
            info.powercaps = {100000, 200000, 250000, 300000, 400000};
            break;
        default:
            // Default to RTX 3090 settings
            info.powercaps = {100000, 200000, 300000, 400000, 450000};
            fprintf(stderr, "Warning: Unknown GPU, using RTX 3090 powercap settings\\n");
    }

    return info;
}

// Configure GPU: set persistent mode and disable other GPUs
void configureGPU() {
    unsigned int deviceCount;
    if (nvmlDeviceGetCount(&deviceCount) != NVML_SUCCESS) {
        fprintf(stderr, "Warning: Failed to get device count\\n");
        return;
    }

    // Enable device 0 and set persistent mode using nvidia-smi
    int ret = system("sudo nvidia-smi -i 0 -pm 1 > /dev/null 2>&1");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to set persistent mode (check passwordless sudo for nvidia-smi)\\n");
    }

    // Set compute mode to exclusive process (mode 3)
    ret = system("sudo nvidia-smi -i 0 -c 3 > /dev/null 2>&1");
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to set exclusive compute mode (check passwordless sudo for nvidia-smi)\\n");
    }

    // Disable other devices if present (set to prohibited mode = 2)
    for (unsigned int i = 1; i < deviceCount; i++) {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "sudo nvidia-smi -i %u -c 2 > /dev/null 2>&1", i);
        ret = system(cmd);
        if (ret != 0) {
            fprintf(stderr, "Warning: Failed to disable GPU %u (check passwordless sudo for nvidia-smi)\\n", i);
        }
    }
}

// Set power limit for the device using nvidia-smi
bool setPowerLimit(nvmlDevice_t device, unsigned int powerLimitMw) {
    unsigned int powerLimitW = powerLimitMw / 1000;
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "sudo nvidia-smi -i 0 -pl %u > /dev/null 2>&1", powerLimitW);

    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Warning: Failed to set power limit to %u W (check passwordless sudo for nvidia-smi)\\n", powerLimitW);
        return false;
    }

    // Wait a bit for the power limit to take effect
    usleep(500000);  // 500ms
    return true;
}

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

struct BenchmarkStats {
    float measurements[3];
    float mean;
    float std_dev;
    float cv;  // Coefficient of Variation
    double gflops;

    // Energy measurements (5 rounds)
    float energy_measurements[5];  // in millijoules
    float energy_mean;
    float energy_std_dev;
    float energy_cv;
    float power;  // in watts (energy_mean / mean_time_ms)
};

template<typename GemmKernel>
BenchmarkStats benchmark_matmul_with_stats(int M, int N, int K) {
    const int warmup_iters = 10;
    const int measured_iters = 100;  // For time measurement
    const int energy_measured_iters = 200;  // For energy measurement
    const int num_rounds = 3;

    BenchmarkStats stats;
    for (int i = 0; i < 3; i++) stats.measurements[i] = 0.0f;
    stats.mean = 0.0f;
    stats.std_dev = 0.0f;
    stats.cv = 0.0f;
    stats.gflops = 0.0;

    // Initialize energy measurements
    for (int i = 0; i < 5; i++) stats.energy_measurements[i] = 0.0f;
    stats.energy_mean = 0.0f;
    stats.energy_std_dev = 0.0f;
    stats.energy_cv = 0.0f;
    stats.power = 0.0f;

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
    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return stats;
    if (gemm_op.initialize(args) != cutlass::Status::kSuccess) return stats;

    // Warmup once before all measurements
    for (int i = 0; i < warmup_iters; ++i) gemm_op();
    cudaDeviceSynchronize();

    // Run 3 rounds of measurements
    for (int round = 0; round < num_rounds; ++round) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < measured_iters; ++i) {
            if (gemm_op() != cutlass::Status::kSuccess) return stats;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        stats.measurements[round] = ms / measured_iters;
    }

    // Calculate statistics
    float sum = 0.0f;
    for (int i = 0; i < num_rounds; ++i) {
        sum += stats.measurements[i];
    }
    stats.mean = sum / num_rounds;

    float variance = 0.0f;
    for (int i = 0; i < num_rounds; ++i) {
        float diff = stats.measurements[i] - stats.mean;
        variance += diff * diff;
    }
    stats.std_dev = std::sqrt(variance / num_rounds);
    stats.cv = (stats.mean > 0) ? (stats.std_dev / stats.mean) : 0.0f;

    // Calculate GFLOPS based on mean time
    stats.gflops = (2.0 * (long long)M * N * K * 1e-9) / (stats.mean * 1e-3);

    // === Energy Measurement ===
    // Get NVML device handle
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) {
        fprintf(stderr, "Warning: Failed to get NVML device handle, skipping energy measurement\\n");
        return stats;
    }

    // Warmup once before all energy measurements
    for (int i = 0; i < warmup_iters; ++i) gemm_op();
    cudaDeviceSynchronize();

    // Run 5 rounds of energy measurements (200 measured iterations per round)
    const int energy_rounds = 5;
    for (int round = 0; round < energy_rounds; ++round) {
        // Get initial energy reading
        unsigned long long energy_start;
        if (nvmlDeviceGetTotalEnergyConsumption(device, &energy_start) != NVML_SUCCESS) {
            fprintf(stderr, "Warning: Failed to get energy reading, skipping energy measurement\\n");
            return stats;
        }

        // Measured iterations
        for (int i = 0; i < energy_measured_iters; ++i) {
            gemm_op();
        }
        cudaDeviceSynchronize();

        // Get final energy reading
        unsigned long long energy_end;
        if (nvmlDeviceGetTotalEnergyConsumption(device, &energy_end) != NVML_SUCCESS) {
            fprintf(stderr, "Warning: Failed to get energy reading, skipping energy measurement\\n");
            return stats;
        }

        // Calculate energy per iteration in millijoules
        stats.energy_measurements[round] = (float)(energy_end - energy_start) / energy_measured_iters;
    }

    // Calculate energy statistics
    float energy_sum = 0.0f;
    for (int i = 0; i < energy_rounds; ++i) {
        energy_sum += stats.energy_measurements[i];
    }
    stats.energy_mean = energy_sum / energy_rounds;

    float energy_variance = 0.0f;
    for (int i = 0; i < energy_rounds; ++i) {
        float diff = stats.energy_measurements[i] - stats.energy_mean;
        energy_variance += diff * diff;
    }
    stats.energy_std_dev = std::sqrt(energy_variance / energy_rounds);
    stats.energy_cv = (stats.energy_mean > 0) ? (stats.energy_std_dev / stats.energy_mean) : 0.0f;

    // Calculate power in watts: energy(mJ) / time(ms) = J/s = W
    stats.power = stats.energy_mean / stats.mean;

    return stats;
}

void ensure_directory_exists(const char* dir) {
    struct stat st;
    if (stat(dir, &st) != 0) {
        mkdir(dir, 0755);
    }
}

void write_csv_header(const char* case_name) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/summary.csv", case_name);
    std::ofstream csv(filename);
    if (csv.is_open()) {
        csv << "id,powercap,M,N,K,tb_m,tb_n,tb_k,warp_m,warp_n,warp_k,stages,time(ms),cv_time,gflops,energy(mj),cv_energy,power\\n";
        csv.close();
    }
}

void append_to_csv(const char* case_name, int config_id, unsigned int powercap_w,
                  int M, int N, int K,
                  int tb_m, int tb_n, int tb_k,
                  int warp_m, int warp_n, int warp_k, int stages,
                  const BenchmarkStats& stats) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/summary.csv", case_name);
    std::ofstream csv(filename, std::ios::app);
    if (csv.is_open()) {
        csv << std::fixed << std::setprecision(3);
        csv << config_id << "," << powercap_w << "," << M << "," << N << "," << K << ","
            << tb_m << "," << tb_n << "," << tb_k << ","
            << warp_m << "," << warp_n << "," << warp_k << "," << stages << ","
            << stats.mean << "," << std::setprecision(4) << stats.cv << ","
            << std::setprecision(0) << stats.gflops << ","
            << std::setprecision(3) << stats.energy_mean << ","
            << std::setprecision(4) << stats.energy_cv << ","
            << (int)(stats.power + 0.5) << "\\n";
        csv.close();
    }
}

void write_config_results(const char* case_name, int config_id, unsigned int powercap_w,
                         int M, int N, int K,
                         int tb_m, int tb_n, int tb_k,
                         int warp_m, int warp_n, int warp_k, int stages,
                         const BenchmarkStats& stats) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/config_%d_pow%d.txt", case_name, config_id, powercap_w);

    std::ofstream out(filename);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open %s\\n", filename);
        return;
    }

    out << std::fixed << std::setprecision(3);

    out << "=== Configuration " << config_id << " (Power Cap: " << powercap_w << "W) ===" << std::endl;
    out << "Problem Size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    out << "Threadblock: " << tb_m << "x" << tb_n << "x" << tb_k << std::endl;
    out << "Warp: " << warp_m << "x" << warp_n << "x" << warp_k << std::endl;
    out << "Stages: " << stages << std::endl;
    out << std::endl;

    out << "=== Measurements (3 rounds of 100 iterations) ===" << std::endl;
    out << "Time per iteration (ms):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        out << "  Round " << (i+1) << ": " << stats.measurements[i] << " ms" << std::endl;
    }
    out << std::endl;

    out << "=== Statistics ===" << std::endl;
    out << "Mean time:                " << stats.mean << " ms" << std::endl;
    out << "Standard deviation:       " << stats.std_dev << " ms" << std::endl;
    out << std::setprecision(4);
    out << "Coefficient of Variation: " << (stats.cv * 100) << "%" << std::endl;
    out << std::endl;

    out << std::setprecision(0);
    out << "=== Performance ===" << std::endl;
    out << "GFLOPS (based on mean):   " << stats.gflops << std::endl;
    out << std::endl;

    out << std::fixed << std::setprecision(3);
    out << "=== Energy Measurements (5 rounds of 200 iterations) ===" << std::endl;
    out << "Energy per iteration (mJ):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        out << "  Round " << (i+1) << ": " << stats.energy_measurements[i] << " mJ" << std::endl;
    }
    out << std::endl;

    out << "=== Energy Statistics ===" << std::endl;
    out << "Mean energy:              " << stats.energy_mean << " mJ" << std::endl;
    out << "Standard deviation:       " << stats.energy_std_dev << " mJ" << std::endl;
    out << std::setprecision(4);
    out << "Coefficient of Variation: " << (stats.energy_cv * 100) << "%" << std::endl;
    out << std::endl;

    out << std::setprecision(3);
    out << "=== Power ===" << std::endl;
    out << "Power (based on mean):    " << stats.power << " W" << std::endl;

    out.close();
}

#define BENCHMARK_CONFIG(DEVICE, POWERCAP_W, CASE_NAME, CONFIG_ID, M, N, K, TBM, TBN, TBK, WM, WN, WK, STAGES) \\
    { \\
        fprintf(stderr, "Running %s config_%d powercap_%dW: tb=%dx%dx%d warp=%dx%dx%d stages=%d\\n", \\
                CASE_NAME, CONFIG_ID, POWERCAP_W, TBM, TBN, TBK, WM, WN, WK, STAGES); \\
        if (!setPowerLimit(DEVICE, POWERCAP_W * 1000)) { \\
            fprintf(stderr, "  -> SKIPPED (failed to set powercap)\\n"); \\
        } else { \\
            using Config = MatMulConfig<TBM, TBN, TBK, WM, WN, WK, STAGES>; \\
            BenchmarkStats stats = benchmark_matmul_with_stats<typename Config::Gemm>(M, N, K); \\
            if (stats.mean > 0) { \\
                write_config_results(CASE_NAME, CONFIG_ID, POWERCAP_W, M, N, K, \\
                                   TBM, TBN, TBK, WM, WN, WK, STAGES, stats); \\
                append_to_csv(CASE_NAME, CONFIG_ID, POWERCAP_W, M, N, K, \\
                             TBM, TBN, TBK, WM, WN, WK, STAGES, stats); \\
                fprintf(stderr, "  -> Time: %.3f ms (CV: %.2f%%), GFLOPS: %.0f, Energy: %d mJ (CV: %.2f%%), Power: %d W\\n", \\
                       stats.mean, stats.cv * 100, stats.gflops, \\
                       (int)(stats.energy_mean + 0.5), stats.energy_cv * 100, (int)(stats.power + 0.5)); \\
            } else { \\
                fprintf(stderr, "  -> FAILED\\n"); \\
            } \\
        } \\
    }

int main() {
    // Initialize NVML
    if (nvmlInit() != NVML_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize NVML\\n");
        return 1;
    }

    // Configure GPU: set persistent mode, disable other GPUs
    configureGPU();

    // Get device 0 handle and GPU info
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) {
        fprintf(stderr, "Error: Failed to get device handle\\n");
        nvmlShutdown();
        return 1;
    }

    GPUInfo gpuInfo = getGPUInfo(device);
    fprintf(stderr, "Detected GPU: %s\\n", gpuInfo.name.c_str());
    fprintf(stderr, "Power cap settings (%zu total): ", gpuInfo.powercaps.size());
    for (size_t i = 0; i < gpuInfo.powercaps.size(); i++) {
        fprintf(stderr, "%dW%s", gpuInfo.powercaps[i] / 1000,
                i < gpuInfo.powercaps.size() - 1 ? ", " : "\\n");
    }
    fprintf(stderr, "\\n");

    // Create output directories and CSV headers
    ensure_directory_exists("case1");
    ensure_directory_exists("case2");
    ensure_directory_exists("case3");
    ensure_directory_exists("case4");

    write_csv_header("case1");
    write_csv_header("case2");
    write_csv_header("case3");
    write_csv_header("case4");

    fprintf(stderr, "Starting multi-size benchmark with energy measurement...\\n");
    fprintf(stderr, "Results will be saved to case{1-4}/summary.csv\\n\\n");

"""

    template_footer = """
    fprintf(stderr, "\\n=== Benchmark completed! ===\\n");
    fprintf(stderr, "Results organized in case{1-4}/ directories:\\n");
    fprintf(stderr, "  - summary.csv: CSV summary of all configs with powercap sweep\\n");
    fprintf(stderr, "  - config_*_pow*.txt: Detailed statistics for each config and powercap\\n");

    // Shutdown NVML
    nvmlShutdown();
    return 0;
}
"""

    with open(output_file, 'w') as f:
        f.write(template_header)

        # Generate benchmark calls for each problem size and config
        configs_to_run = configs[:2] if test_mode else configs

        for case_name, (M, N, K) in problem_sizes.items():
            f.write(f"\n    // {case_name}: M={M}, N={N}, K={K}\n")
            if test_mode:
                f.write(f"    // TEST MODE: Running only first 2 configs with all powercaps\n")
            else:
                f.write(f"    // Running all {len(configs_to_run)} configs with all powercaps\n")

            for i, cfg in enumerate(configs_to_run):
                f.write(f"    for (auto powercap_mw : gpuInfo.powercaps) {{\n")
                f.write(f"        unsigned int powercap_w = powercap_mw / 1000;\n")
                f.write(f"        BENCHMARK_CONFIG(device, powercap_w, \"{case_name}\", {i}, {M}, {N}, {K}, "
                       f"{cfg['tb_m']}, {cfg['tb_n']}, {cfg['tb_k']}, "
                       f"{cfg['warp_m']}, {cfg['warp_n']}, {cfg['warp_k']}, "
                       f"{cfg['stages']});\n")
                f.write(f"    }}\n")

        f.write(template_footer)

    num_configs = len(configs_to_run)
    num_powercaps = 5  # All GPUs have 5 powercap settings
    total_benchmarks = num_configs * len(problem_sizes) * num_powercaps

    if test_mode:
        print(f"Generated TEST MODE benchmark code")
        print(f"Running first {num_configs} configs × {num_powercaps} powercaps × {len(problem_sizes)} problem sizes")
    else:
        print(f"Generated benchmark code with {num_configs} configs × {num_powercaps} powercaps × {len(problem_sizes)} problem sizes")

    print(f"Total benchmarks: {total_benchmarks}")
    print(f"Each benchmark:")
    print(f"  - Time: 10 warmup + 3 rounds × 100 iterations")
    print(f"  - Energy: 10 warmup + 5 rounds × 200 iterations")

    if test_mode:
        print(f"\nEstimated runtime: ~5-8 minutes ({total_benchmarks} benchmarks total)")
    else:
        print(f"\nEstimated runtime: ~2-3 hours ({total_benchmarks} benchmarks total)")

    return total_benchmarks

if __name__ == "__main__":
    import sys

    # Check for test mode flag
    test_mode = '--test' in sys.argv

    configs = read_configs('configs')
    print(f"Read {len(configs)} configurations from 'configs' file")

    problem_sizes = {
        'case1': (8192, 8192, 8192),
        'case2': (8192, 8192, 16384),
        'case3': (8192, 8192, 4096),
        'case4': (16384, 16384, 1024)
    }

    output_file = 'multisize_benchmark_test.cu' if test_mode else 'multisize_benchmark.cu'
    generate_benchmark_code(configs, problem_sizes, output_file, test_mode=test_mode)

    print("\nProblem sizes:")
    for case, (M, N, K) in problem_sizes.items():
        print(f"  {case}: M={M}, N={N}, K={K}")

    if test_mode:
        print("\nTo compile and run TEST MODE:")
        print("  make multisize-test")
        print("  sudo ./multisize_benchmark_test  # Requires sudo for powercap control")
    else:
        print("\nTo compile and run:")
        print("  make multisize")
        print("  sudo ./multisize_benchmark  # Requires sudo for powercap control")

    print("\nOutput structure:")
    print("  case1/summary.csv         - CSV summary (id,powercap,M,N,K,tb_m,...,power)")
    print("  case1/config_0_pow100.txt - Detailed stats for config 0 at 100W")
    print("  case1/config_0_pow200.txt - Detailed stats for config 0 at 200W")
    print("  ... (same for case2/, case3/, case4/)")
    print("\nNOTE: Setting power caps requires root privileges.")
