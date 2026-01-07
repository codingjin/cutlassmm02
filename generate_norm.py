#!/usr/bin/env python3
"""
Generate normalized CSV files from summary.csv in each case folder.

Reads case{1-4}/summary.csv and generates case{1-4}/norm.csv with:
- id, power, gflops, time, energy, EDP
- norm_time, norm_energy, norm_mul, norm_add
"""

import csv
import os
import sys

def process_case(case_name):
    """Process a single case folder to generate norm.csv from summary.csv"""

    summary_file = f"{case_name}/summary.csv"
    norm_file = f"{case_name}/norm.csv"

    # Check if summary.csv exists
    if not os.path.exists(summary_file):
        print(f"Warning: {summary_file} not found, skipping...")
        return

    # Read summary.csv
    rows = []
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'id': int(row['id']),
                'power': int(row['powercap']),
                'gflops': float(row['gflops']),
                'time': float(row['time(ms)']),
                'energy': float(row['energy(mj)'])
            })

    if len(rows) == 0:
        print(f"Warning: No data in {summary_file}, skipping...")
        return

    # Find min and max time and energy
    min_time = min(row['time'] for row in rows)
    max_time = max(row['time'] for row in rows)
    min_energy = min(row['energy'] for row in rows)
    max_energy = max(row['energy'] for row in rows)

    print(f"{case_name}:")
    print(f"  Time range: {min_time:.3f} - {max_time:.3f} ms")
    print(f"  Energy range: {min_energy:.3f} - {max_energy:.3f} mJ")

    # Calculate normalized metrics for each row
    for row in rows:
        # EDP = time * energy
        row['EDP'] = row['time'] * row['energy']

        # Normalized time
        if max_time == min_time:
            row['norm_time'] = 0.0
        else:
            row['norm_time'] = (row['time'] - min_time) / (max_time - min_time)

        # Normalized energy
        if max_energy == min_energy:
            row['norm_energy'] = 0.0
        else:
            row['norm_energy'] = (row['energy'] - min_energy) / (max_energy - min_energy)

        # norm_mul = norm_time * norm_energy
        row['norm_mul'] = row['norm_time'] * row['norm_energy']

        # norm_add = norm_time + norm_energy
        row['norm_add'] = row['norm_time'] + row['norm_energy']

    # Write norm.csv
    with open(norm_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['id', 'power', 'gflops', 'time', 'energy', 'EDP',
                        'norm_time', 'norm_energy', 'norm_mul', 'norm_add'])

        # Write data rows
        for row in rows:
            writer.writerow([
                row['id'],
                row['power'],
                f"{row['gflops']:.0f}",
                f"{row['time']:.3f}",
                f"{row['energy']:.3f}",
                f"{row['EDP']:.3f}",
                f"{row['norm_time']:.3g}",
                f"{row['norm_energy']:.3g}",
                f"{row['norm_mul']:.3g}",
                f"{row['norm_add']:.3g}"
            ])

    print(f"  Generated {norm_file} with {len(rows)} rows\n")

def main():
    """Process all case folders"""
    cases = ['case1', 'case2', 'case3', 'case4']

    print("Generating normalized CSV files...\n")

    for case in cases:
        process_case(case)

    print("Done!")

if __name__ == '__main__':
    main()
