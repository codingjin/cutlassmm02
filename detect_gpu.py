#!/usr/bin/env python3
"""Detect GPU and provide configuration parameters"""

import subprocess
import re
import sys

class GPUConfig:
    def __init__(self, name, compute_cap, arch_flag, peak_tflops_tf32, shared_mem_per_sm_kb):
        self.name = name
        self.compute_cap = compute_cap
        self.arch_flag = arch_flag
        self.peak_tflops_tf32 = peak_tflops_tf32
        self.shared_mem_per_sm_kb = shared_mem_per_sm_kb

    def __str__(self):
        return f"{self.name} (SM{self.compute_cap.replace('.', '')})"

# GPU specifications database
GPU_SPECS = {
    'A100': GPUConfig('A100', '8.0', 'sm_80', 156, 164),      # 156 TFLOPS TF32, 164 KB shared mem/SM
    'RTX 3090': GPUConfig('RTX 3090', '8.6', 'sm_86', 71, 100),  # 71 TFLOPS TF32, 100 KB shared mem/SM
    'RTX 4090': GPUConfig('RTX 4090', '8.9', 'sm_89', 165, 100), # 165 TFLOPS TF32, 100 KB shared mem/SM
}

def detect_gpu():
    """Detect current GPU using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        if not output:
            return None

        # Parse output: "NVIDIA GeForce RTX 3090, 8.6"
        parts = output.split(',')
        if len(parts) < 2:
            return None

        gpu_name = parts[0].strip()
        compute_cap = parts[1].strip()

        # Match against known GPUs
        for key, config in GPU_SPECS.items():
            if key in gpu_name and config.compute_cap == compute_cap:
                return config

        # Fallback: try to create config from compute capability
        if compute_cap in ['8.0', '8.6', '8.9']:
            print(f"Warning: Unknown GPU '{gpu_name}' with compute {compute_cap}", file=sys.stderr)
            print(f"Using default settings for SM{compute_cap.replace('.', '')}", file=sys.stderr)

            # Use closest match
            if compute_cap == '8.0':
                return GPU_SPECS['A100']
            elif compute_cap == '8.6':
                return GPU_SPECS['RTX 3090']
            elif compute_cap == '8.9':
                return GPU_SPECS['RTX 4090']

        return None

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_gpu_config(override=None):
    """Get GPU configuration, with optional manual override"""
    if override:
        if override in GPU_SPECS:
            return GPU_SPECS[override]
        else:
            print(f"Error: Unknown GPU '{override}'", file=sys.stderr)
            print(f"Supported GPUs: {', '.join(GPU_SPECS.keys())}", file=sys.stderr)
            sys.exit(1)

    # Auto-detect
    config = detect_gpu()
    if config:
        return config

    # Fallback to RTX 3090 if detection fails
    print("Warning: Could not detect GPU, defaulting to RTX 3090", file=sys.stderr)
    return GPU_SPECS['RTX 3090']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Detect GPU and show configuration')
    parser.add_argument('--gpu', choices=list(GPU_SPECS.keys()),
                       help='Override GPU selection')
    parser.add_argument('--format', choices=['human', 'makefile', 'json'],
                       default='human',
                       help='Output format')

    args = parser.parse_args()

    config = get_gpu_config(args.gpu)

    if args.format == 'human':
        print(f"GPU: {config.name}")
        print(f"Compute Capability: {config.compute_cap}")
        print(f"Architecture Flag: -{config.arch_flag}")
        print(f"Peak TF32 TFLOPS: {config.peak_tflops_tf32}")
        print(f"Shared Memory/SM: {config.shared_mem_per_sm_kb} KB")

    elif args.format == 'makefile':
        # Output as shell variable exports that can be sourced
        print(f"export GPU_ARCH={config.arch_flag}")
        print(f"export GPU_NAME=\"{config.name}\"")
        print(f"export PEAK_TFLOPS={config.peak_tflops_tf32}")
        print(f"export SMEM_PER_SM_KB={config.shared_mem_per_sm_kb}")

    elif args.format == 'json':
        import json
        print(json.dumps({
            'name': config.name,
            'compute_cap': config.compute_cap,
            'arch_flag': config.arch_flag,
            'peak_tflops_tf32': config.peak_tflops_tf32,
            'shared_mem_per_sm_kb': config.shared_mem_per_sm_kb
        }, indent=2))
