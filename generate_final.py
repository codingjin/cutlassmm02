#!/usr/bin/env python3
"""
Generate final.csv from norm.csv by combining rows with the same id.

Reads case{1-4}/norm.csv and generates case{1-4}/final.csv where:
- Each unique id gets one row
- Power column is removed
- All metrics for different power levels are combined into one row
"""

import csv
import os
from collections import defaultdict

def process_case(case_name):
    """Process a single case folder to generate final.csv from norm.csv"""

    norm_file = f"{case_name}/norm.csv"
    final_file = f"{case_name}/final.csv"

    # Check if norm.csv exists
    if not os.path.exists(norm_file):
        print(f"Warning: {norm_file} not found, skipping...")
        return

    # Read norm.csv and group by id
    data_by_id = defaultdict(list)

    with open(norm_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config_id = int(row['id'])
            power = int(row['power'])

            # Extract metrics (excluding id and power)
            metrics = {
                'power': power,  # Keep for sorting
                'gflops': row['gflops'],
                'time': row['time'],
                'energy': row['energy'],
                'EDP': row['EDP'],
                'norm_time': row['norm_time'],
                'norm_energy': row['norm_energy'],
                'norm_mul': row['norm_mul'],
                'norm_add': row['norm_add']
            }

            data_by_id[config_id].append(metrics)

    if len(data_by_id) == 0:
        print(f"Warning: No data in {norm_file}, skipping...")
        return

    # Sort each id's data by power level
    for config_id in data_by_id:
        data_by_id[config_id].sort(key=lambda x: x['power'])

    # Determine number of power levels (should be 5)
    num_power_levels = len(data_by_id[list(data_by_id.keys())[0]])

    print(f"{case_name}:")
    print(f"  Found {len(data_by_id)} unique configs")
    print(f"  Each config has {num_power_levels} power levels")

    # Generate header
    # id + (8 metrics × num_power_levels)
    header = ['id']
    metric_names = ['gflops', 'time', 'energy', 'EDP', 'norm_time', 'norm_energy', 'norm_mul', 'norm_add']

    for _ in range(num_power_levels):
        header.extend(metric_names)

    # Write final.csv
    with open(final_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Write data rows (sorted by id)
        for config_id in sorted(data_by_id.keys()):
            row = [config_id]

            # Add metrics for each power level in order
            for power_data in data_by_id[config_id]:
                for metric_name in metric_names:
                    row.append(power_data[metric_name])

            writer.writerow(row)

    print(f"  Generated {final_file} with {len(data_by_id)} rows\n")

def main():
    """Process all case folders"""
    cases = ['case1', 'case2', 'case3', 'case4']

    print("Generating final CSV files...\n")

    for case in cases:
        process_case(case)

    print("Done!")

if __name__ == '__main__':
    main()
