#!/bin/bash
# Complete benchmark workflow: configs → final.csv → final0.csv
# Usage: ./run_complete_benchmark.sh [--test|--full]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Parse arguments
MODE="full"  # Default to full mode
if [ "$1" == "--full" ]; then
    MODE="full"
elif [ "$1" == "--test" ]; then
    MODE="test"
elif [ -n "$1" ]; then
    print_error "Unknown option: $1"
    echo "Usage: $0 [--test|--full]"
    echo "  --test: Run test mode (2 configs × 5 powercaps × 4 cases = 40 benchmarks, ~5-8 min)"
    echo "  --full: Run full mode (36 configs × 5 powercaps × 4 cases = 720 benchmarks, ~4-5 hours) [DEFAULT]"
    exit 1
fi

print_header "CUTLASS Benchmark Complete Workflow"
echo ""
print_info "Mode: ${MODE^^}"
if [ "$MODE" == "test" ]; then
    print_info "Benchmarks: 2 configs × 5 powercaps × 4 cases = 40 total"
    print_info "Estimated time: ~5-8 minutes"
else
    print_info "Benchmarks: 36 configs × 5 powercaps × 4 cases = 720 total"
    print_info "Estimated time: ~4-5 hours"
fi
echo ""

# ============================================
# STEP 1: Check prerequisites
# ============================================
print_header "Step 1/7: Checking Prerequisites"

# Check if configs file exists
if [ ! -f "configs" ]; then
    print_error "configs file not found!"
    echo "Please create a 'configs' file with kernel configurations."
    echo "Format: tb_m tb_n tb_k warp_m warp_n warp_k stages"
    exit 1
fi

NUM_CONFIGS=$(wc -l < configs)
print_success "Found configs file with $NUM_CONFIGS configurations"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found! Please install CUDA."
    exit 1
fi
print_success "nvidia-smi found"

# Check if passwordless sudo works for nvidia-smi
if sudo -n nvidia-smi -i 0 -q > /dev/null 2>&1; then
    print_success "Passwordless sudo for nvidia-smi is configured"
else
    print_warning "Passwordless sudo for nvidia-smi not configured"
    echo ""
    echo "To enable passwordless sudo for nvidia-smi, run:"
    echo ""
    echo "sudo bash -c 'cat > /etc/sudoers.d/nvidia-smi << EOF"
    echo "'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi"
    echo "'$USER' ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi *"
    echo "EOF'"
    echo ""
    echo "sudo chmod 0440 /etc/sudoers.d/nvidia-smi"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i 0 2>/dev/null || echo "Unknown")
print_info "Detected GPU: $GPU_NAME"

echo ""

# ============================================
# STEP 2: Clean previous results
# ============================================
print_header "Step 2/7: Cleaning Previous Results"

if [ -d "case1" ] || [ -d "case2" ] || [ -d "case3" ] || [ -d "case4" ]; then
    print_warning "Found existing results in case* directories"
    read -p "Delete and start fresh? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        make clean > /dev/null 2>&1
        print_success "Cleaned previous results"
    else
        print_info "Keeping existing results (may be overwritten)"
    fi
else
    print_info "No previous results found"
fi

echo ""

# ============================================
# STEP 3: Build benchmark
# ============================================
print_header "Step 3/7: Building Benchmark"

if [ "$MODE" == "test" ]; then
    print_info "Building test mode benchmark..."
    if make multisize-test 2>&1 | grep -q "Build complete"; then
        print_success "Test benchmark compiled successfully"
        BENCHMARK_BIN="./multisize_benchmark_test"
    else
        print_error "Compilation failed!"
        exit 1
    fi
else
    print_info "Building full benchmark (this may take 1-2 minutes)..."
    if make multisize 2>&1 | grep -q "Build complete"; then
        print_success "Full benchmark compiled successfully"
        BENCHMARK_BIN="./multisize_benchmark"
    else
        print_error "Compilation failed!"
        exit 1
    fi
fi

echo ""

# ============================================
# STEP 4: Run benchmark
# ============================================
print_header "Step 4/7: Running Benchmark"

if [ "$MODE" == "test" ]; then
    print_info "Running 40 benchmarks (2 configs × 5 powercaps × 4 cases)..."
else
    print_info "Running 720 benchmarks (36 configs × 5 powercaps × 4 cases)..."
    print_warning "This will take approximately 4-5 hours. You can monitor progress in real-time."
fi

echo ""
START_TIME=$(date +%s)

# Run benchmark and capture output
if $BENCHMARK_BIN 2>&1 | tee benchmark_output.log; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    echo ""
    print_success "Benchmark completed in ${MINUTES}m ${SECONDS}s"
else
    print_error "Benchmark failed! Check benchmark_output.log for details"
    exit 1
fi

echo ""

# ============================================
# STEP 5: Generate normalized metrics
# ============================================
print_header "Step 5/7: Generating Normalized Metrics"

print_info "Processing summary.csv files..."

if make norm 2>&1 | grep -q "Done"; then
    print_success "Generated norm.csv files"

    # Show summary of generated files
    for case in case1 case2 case3 case4; do
        if [ -f "$case/norm.csv" ]; then
            ROWS=$(tail -n +2 "$case/norm.csv" | wc -l)
            print_info "  $case/norm.csv: $ROWS rows"
        fi
    done
else
    print_error "Failed to generate normalized metrics"
    exit 1
fi

echo ""

# ============================================
# STEP 6: Generate final pivoted data
# ============================================
print_header "Step 6/7: Generating Final CSV Files"

print_info "Creating final.csv files (one row per config)..."

if make final 2>&1 | grep -q "Done"; then
    print_success "Generated final.csv files"

    echo ""
    print_header "Results Summary"
    echo ""

    # Show summary of all generated files
    for case in case1 case2 case3 case4; do
        if [ -f "$case/final.csv" ]; then
            ROWS=$(tail -n +2 "$case/final.csv" | wc -l)
            COLS=$(head -1 "$case/final.csv" | awk -F',' '{print NF}')
            print_success "$case/final.csv: $ROWS configs × $COLS columns"
        fi
    done

    echo ""
    print_info "All output files per case:"
    echo "  - summary.csv: Raw measurements (time, energy, GFLOPS, power)"
    echo "  - norm.csv:    Normalized metrics (EDP, norm_time, norm_energy, etc.)"
    echo "  - final.csv:   Pivoted data (one row per config with all power levels)"
    echo "  - final0.csv:  Pivoted data (removes gflops and norm_mul columns)"

else
    print_error "Failed to generate final CSV files"
    exit 1
fi

echo ""

# ============================================
# STEP 7: Generate final0 CSV files
# ============================================
print_header "Step 7/7: Generating Final0 CSV Files"

print_info "Creating final0.csv files (removing gflops and norm_mul columns)..."

if make final0 2>&1 | grep -q "Done"; then
    print_success "Generated final0.csv files"

    # Show summary of generated files
    for case in case1 case2 case3 case4; do
        if [ -f "$case/final0.csv" ]; then
            ROWS=$(tail -n +2 "$case/final0.csv" | wc -l)
            COLS=$(head -1 "$case/final0.csv" | awk -F',' '{print NF}')
            print_info "  $case/final0.csv: $ROWS configs × $COLS columns"
        fi
    done
else
    print_error "Failed to generate final0 CSV files"
    exit 1
fi

echo ""

# ============================================
# Final summary
# ============================================
print_header "Workflow Complete!"
echo ""
print_success "All final.csv and final0.csv files are ready!"
echo ""
echo "View results:"
echo "  ls -lh case*/final.csv case*/final0.csv"
echo "  head case1/final.csv"
echo "  head case1/final0.csv"
echo ""
echo "Preview in column format:"
echo "  column -t -s',' case1/final.csv | less -S"
echo "  column -t -s',' case1/final0.csv | less -S"
echo ""
echo "Full results location:"
for case in case1 case2 case3 case4; do
    if [ -f "$case/final.csv" ]; then
        echo "  $(pwd)/$case/final.csv"
    fi
    if [ -f "$case/final0.csv" ]; then
        echo "  $(pwd)/$case/final0.csv"
    fi
done
echo ""

# Show benchmark log location
if [ -f "benchmark_output.log" ]; then
    print_info "Full benchmark output saved to: benchmark_output.log"
fi

echo ""
print_success "Done! 🎉"
