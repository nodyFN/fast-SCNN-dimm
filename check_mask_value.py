#!/usr/bin/env python3
"""
Scan dataset masks and report value distribution.

Usage
-----
::

    python check_mask_value.py --data-root duts_data
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check mask value distribution in dataset"
    )
    parser.add_argument("--data-root", type=str, default="duts_data")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    mask_dir = Path(args.data_root) / args.split / "masks"
    if not mask_dir.exists():
        print(f"Mask directory not found: {mask_dir}")
        return

    mask_files = sorted(mask_dir.glob("*.png"))[:args.max_samples]
    print(f"Scanning {len(mask_files)} masks in {mask_dir}...")

    value_counter = Counter()
    stats = {"binary_01": 0, "binary_0255": 0, "grayscale": 0}

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  Warning: Failed to load {mask_path.name}")
            continue

        unique = np.unique(mask)
        for v in unique:
            value_counter[int(v)] += 1

        if set(unique).issubset({0, 1}):
            stats["binary_01"] += 1
        elif set(unique).issubset({0, 255}):
            stats["binary_0255"] += 1
        else:
            stats["grayscale"] += 1

    print(f"\nResults ({len(mask_files)} masks scanned):")
    print(f"  Binary {{0, 1}}:   {stats['binary_01']}")
    print(f"  Binary {{0, 255}}: {stats['binary_0255']}")
    print(f"  Grayscale:        {stats['grayscale']}")
    print(f"\nUnique values found: {sorted(value_counter.keys())}")
    if stats["grayscale"] > 0:
        print("\n⚠ Some masks contain grayscale values. "
              "Use --allow-threshold in training to handle them.")


if __name__ == "__main__":
    main()
