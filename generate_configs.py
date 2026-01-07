#!/usr/bin/env python3
"""Generate all valid configurations and save to file"""

import sys
sys.path.insert(0, '.')
from autotune import generate_search_space
from detect_gpu import get_gpu_config

if __name__ == "__main__":
    # Detect or override GPU
    gpu_override = None
    if '--gpu' in sys.argv:
        idx = sys.argv.index('--gpu')
        if idx + 1 < len(sys.argv):
            gpu_override = sys.argv[idx + 1]

    gpu_config = get_gpu_config(gpu_override)
    print(f"Generating configs for: {gpu_config.name}")

    configs = generate_search_space(gpu_config)

    with open('configs', 'w') as f:
        f.write(f"# Target GPU: {gpu_config.name} ({gpu_config.arch_flag})\n")
        f.write(f"# Total valid configurations: {len(configs)}\n")
        f.write(f"# Format: tb_m,tb_n,tb_k,warp_m,warp_n,warp_k,stages\n")
        f.write("#" + "="*70 + "\n\n")

        for i, config in enumerate(configs, 1):
            f.write(f"{config.tb_m},{config.tb_n},{config.tb_k},"
                   f"{config.warp_m},{config.warp_n},{config.warp_k},"
                   f"{config.stages}\n")

    print(f"Generated {len(configs)} configurations in 'configs' file")
