#!/usr/bin/env python3
"""
Generate final0.csv files from final.csv files.
Removes all 'gflops' and 'norm_mul' columns.
"""

import csv
import os
from pathlib import Path


def process_final_csv(input_path: Path, output_path: Path):
    """Process a final.csv file and create final0.csv without gflops and norm_mul columns."""

    with open(input_path, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        # Find indices of columns to keep (exclude 'gflops' and 'norm_mul')
        keep_indices = []
        kept_headers = []

        for i, col in enumerate(header):
            # Keep all columns except 'gflops' and 'norm_mul'
            if col not in ['gflops', 'norm_mul']:
                keep_indices.append(i)
                kept_headers.append(col)

        # Write output file
        with open(output_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(kept_headers)

            # Process data rows
            for row in reader:
                filtered_row = [row[i] for i in keep_indices]
                writer.writerow(filtered_row)

    return len(kept_headers)


def main():
    """Process all final.csv files in case1-4 directories."""

    cases = ['case1', 'case2', 'case3', 'case4']

    print("Processing final.csv files...")
    print("=" * 60)

    for case in cases:
        case_dir = Path(case)
        input_file = case_dir / 'final.csv'
        output_file = case_dir / 'final0.csv'

        if not input_file.exists():
            print(f"⚠️  {case}/final.csv not found, skipping...")
            continue

        # Process the file
        num_cols = process_final_csv(input_file, output_file)

        # Count rows
        with open(output_file, 'r') as f:
            num_rows = sum(1 for line in f) - 1  # Exclude header

        print(f"✓ {case}/final0.csv created: {num_rows} rows × {num_cols} columns")

    print("=" * 60)
    print("Done!")
    print()
    print("Columns removed: 'gflops' and 'norm_mul'")
    print("Columns kept: id, time, energy, EDP, norm_time, norm_energy, norm_add")


if __name__ == '__main__':
    main()
