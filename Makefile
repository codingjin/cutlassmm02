# CUTLASS Matrix Multiplication Auto-Tuning Makefile
# Supports: RTX 3090 (SM86), RTX 4090 (SM89), A100 (SM80)

# CUDA and CUTLASS paths
CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH ?= /home/jin/cutlass

# GPU detection and configuration
# Override with: make GPU=A100 or make GPU="RTX 4090"
GPU ?= auto
GPU_DETECT_CMD = python3 detect_gpu.py $(if $(filter-out auto,$(GPU)),--gpu "$(GPU)")
GPU_ARCH := $(shell $(GPU_DETECT_CMD) --format makefile | grep GPU_ARCH | cut -d= -f2)
GPU_NAME := $(shell $(GPU_DETECT_CMD) | head -1 | cut -d: -f2 | xargs)
PEAK_TFLOPS := $(shell $(GPU_DETECT_CMD) --format makefile | grep PEAK_TFLOPS | cut -d= -f2)
SMEM_PER_SM_KB := $(shell $(GPU_DETECT_CMD) --format makefile | grep SMEM_PER_SM_KB | cut -d= -f2)

# Auto-detected or overridden values:
# GPU_ARCH, GPU_NAME, PEAK_TFLOPS, SMEM_PER_SM_KB

# Compiler
NVCC = $(CUDA_PATH)/bin/nvcc

# Compiler flags
NVCC_FLAGS = -std=c++17 \
             -O3 \
             -arch=$(GPU_ARCH) \
             -I$(CUTLASS_PATH)/include \
             -I$(CUTLASS_PATH)/tools/util/include \
             --expt-relaxed-constexpr \
             -Xcompiler=-fPIC

# cuBLAS linking flags
CUBLAS_FLAGS = -lcublas

# NVML linking flags (for energy measurement)
NVML_FLAGS = -lnvidia-ml

# Targets
TARGETS = cutlass_matmul_tuning verify_correctness

.PHONY: all clean autotune run help multisize run-multisize multisize-test run-multisize-test gpu-info norm final

# Show GPU information
gpu-info:
	@echo "======================================"
	@echo "GPU Configuration"
	@echo "======================================"
	@python3 detect_gpu.py
	@echo ""
	@echo "To override GPU: make GPU=A100 <target>"
	@echo "  Example: make GPU=A100 multisize"
	@echo ""

all: gpu-info $(TARGETS)

# Basic auto-tuning executable
cutlass_matmul_tuning: cutlass_matmul_tuning.cu
	@echo "Compiling CUTLASS matrix multiplication auto-tuner..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "Build complete: ./$@"

# Correctness verification executable
verify_correctness: verify_correctness.cu
	@echo "Compiling CUTLASS correctness verification..."
	$(NVCC) $(NVCC_FLAGS) $(CUBLAS_FLAGS) $< -o $@
	@echo "Build complete: ./$@"

# Generate and compile extensive auto-tuning
autotune: autotune.py
	@echo "Generating extensive auto-tuning code..."
	python3 autotune.py
	@echo "Compiling generated auto-tuner..."
	$(NVCC) $(NVCC_FLAGS) cutlass_autotune_generated.cu -o cutlass_autotune_generated
	@echo "Build complete: ./cutlass_autotune_generated"

# Generate and compile multi-size benchmark
multisize: configs generate_multisize_benchmark.py
	@echo "Generating multi-size benchmark code..."
	python3 generate_multisize_benchmark.py
	@echo "Compiling multi-size benchmark (36 configs × 4 problem sizes = 144 benchmarks)..."
	@echo "This may take 1-2 minutes..."
	$(NVCC) $(NVCC_FLAGS) $(NVML_FLAGS) multisize_benchmark.cu -o multisize_benchmark
	@echo "Build complete: ./multisize_benchmark"

# Generate and compile multi-size benchmark (TEST MODE - first 2 configs only)
multisize-test: configs generate_multisize_benchmark.py
	@echo "Generating multi-size benchmark code (TEST MODE)..."
	python3 generate_multisize_benchmark.py --test
	@echo "Compiling test benchmark (2 configs × 4 problem sizes)..."
	$(NVCC) $(NVCC_FLAGS) $(NVML_FLAGS) multisize_benchmark_test.cu -o multisize_benchmark_test
	@echo "Build complete: ./multisize_benchmark_test"

# Run basic tuning
run: cutlass_matmul_tuning
	@echo "Running auto-tuning..."
	./cutlass_matmul_tuning

# Run correctness verification
verify: verify_correctness
	@echo "Running correctness verification..."
	./verify_correctness

# Run extensive tuning and save results
run-autotune: autotune
	@echo "Running extensive auto-tuning (this may take a while)..."
	./cutlass_autotune_generated 2>&1 | tee results.csv
	@echo ""
	@echo "Results saved to results.csv"

# Run multi-size benchmark and save results
run-multisize: multisize
	@echo "Running multi-size benchmark with energy measurement (144 total benchmarks)..."
	@echo "Problem sizes: case1(8192³), case2(8192²×16384), case3(8192²×4096), case4(16384²×1024)"
	@echo "Time: 10 warmup + 3 rounds × 100 iterations"
	@echo "Energy: 10 warmup + 5 rounds × 100 iterations"
	@echo "This may take 15-20 minutes..."
	./multisize_benchmark
	@echo ""
	@echo "Results saved to:"
	@echo "  case1/summary.csv, case2/summary.csv, case3/summary.csv, case4/summary.csv"
	@echo "  case{1-4}/config_*.txt (detailed per-config statistics with energy data)"

# Run multi-size benchmark in TEST MODE (first 2 configs only)
run-multisize-test: multisize-test
	@echo "Running multi-size benchmark with energy measurement in TEST MODE..."
	@echo "Testing first 2 configs × 5 powercaps × 4 problem sizes = 40 benchmarks"
	@echo "Problem sizes: case1(8192³), case2(8192²×16384), case3(8192²×4096), case4(16384²×1024)"
	@echo "Time: 10 warmup + 3 rounds × 100 iterations"
	@echo "Energy: 10 warmup + 5 rounds × 200 iterations"
	@echo "This may take 5-8 minutes..."
	./multisize_benchmark_test
	@echo ""
	@echo "Test results saved to:"
	@echo "  case1/summary.csv (first 2 configs with energy data)"
	@echo "  case2/summary.csv, case3/summary.csv, case4/summary.csv"

# Generate normalized CSV files from summary.csv
norm:
	@echo "Generating normalized CSV files from summary.csv..."
	python3 generate_norm.py
	@echo ""
	@echo "Normalized results saved to:"
	@echo "  case1/norm.csv, case2/norm.csv, case3/norm.csv, case4/norm.csv"

# Generate final CSV files from norm.csv (combines power levels)
final:
	@echo "Generating final CSV files from norm.csv..."
	python3 generate_final.py
	@echo ""
	@echo "Final results saved to:"
	@echo "  case1/final.csv, case2/final.csv, case3/final.csv, case4/final.csv"

# Clean build artifacts
clean:
	rm -f $(TARGETS) cutlass_autotune_generated multisize_benchmark multisize_benchmark_test query_gpu_smem
	rm -f cutlass_autotune_generated.cu multisize_benchmark.cu multisize_benchmark_test.cu
	rm -f results.csv tuning.log
	rm -rf case1 case2 case3 case4
	@echo "Clean complete"

# Help
help:
	@echo "CUTLASS Matrix Multiplication Auto-Tuning"
	@echo "=========================================="
	@echo "Supports: RTX 3090, RTX 4090, A100"
	@echo ""
	@echo "Targets:"
	@echo "  make                     - Build basic auto-tuner and verification"
	@echo "  make gpu-info            - Show detected GPU configuration"
	@echo "  make autotune            - Generate and build extensive auto-tuner"
	@echo "  make multisize           - Generate and build multi-size benchmark (36 configs × 4 sizes)"
	@echo "  make multisize-test      - Generate and build test benchmark (2 configs × 4 sizes)"
	@echo "  make run                 - Run basic auto-tuning (12 configs)"
	@echo "  make run-autotune        - Run extensive auto-tuning (36 configs)"
	@echo "  make run-multisize       - Run multi-size benchmark with energy (144 benchmarks, ~15-20 min)"
	@echo "  make run-multisize-test  - Run test benchmark with energy (40 benchmarks, ~5-8 min)"
	@echo "  make norm                - Generate normalized CSV files from summary.csv"
	@echo "  make final               - Generate final CSV files from norm.csv (combines power levels)"
	@echo "  make verify              - Run correctness verification (vs cuBLAS)"
	@echo "  make clean               - Remove build artifacts"
	@echo "  make help                - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_PATH=$(CUDA_PATH)"
	@echo "  CUTLASS_PATH=$(CUTLASS_PATH)"
	@echo "  GPU=auto (detected: $(GPU_NAME))"
	@echo ""
	@echo "GPU Override:"
	@echo "  make GPU=A100 <target>        - Force A100 configuration"
	@echo "  make GPU=\"RTX 4090\" <target>  - Force RTX 4090 configuration"
	@echo "  make GPU=\"RTX 3090\" <target>  - Force RTX 3090 configuration"
	@echo ""
	@echo "Note: Set CUTLASS_PATH if CUTLASS is installed elsewhere"
	@echo "Example: make CUTLASS_PATH=/path/to/cutlass GPU=A100 multisize"
