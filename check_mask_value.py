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
        description="Check mask value distribution and class count in dataset"
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root dataset directory containing train/masks, etc.")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Direct path to masks folder")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to scan (default: scan all)")
    args = parser.parse_args()

    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
    elif args.data_root:
        mask_dir = Path(args.data_root) / args.split / "masks"
    else:
        mask_dir = Path("duts_data") / args.split / "masks"

    if not mask_dir.exists():
        print(f"Error: Mask directory not found: {mask_dir}")
        return

    all_files = sorted(mask_dir.glob("*.png"))
    if not all_files:
        all_files = sorted(mask_dir.glob("*.jpg")) + sorted(mask_dir.glob("*.jpeg"))

    mask_files = all_files[:args.max_samples] if args.max_samples else all_files
    print(f"Scanning {len(mask_files):,} masks in {mask_dir}...")

    unique_classes = set()
    stats = {"binary_01": 0, "binary_0255": 0, "multiclass": 0}

    for i, mask_path in enumerate(mask_files, start=1):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        unq = np.unique(mask)
        unique_classes.update(int(v) for v in unq)

        unq_set = set(unq)
        if unq_set.issubset({0, 1}):
            stats["binary_01"] += 1
        elif unq_set.issubset({0, 255}):
            stats["binary_0255"] += 1
        else:
            stats["multiclass"] += 1

        if i % 1000 == 0 or i == len(mask_files):
            print(f"  Processed {i:,}/{len(mask_files):,} masks...")

    sorted_classes = sorted(unique_classes)
    valid_classes = [c for c in sorted_classes if c != 255]
    max_class_id = max(valid_classes) if valid_classes else (max(sorted_classes) if sorted_classes else 0)

    print("\n" + "=" * 60)
    print("Mask Class & Value Summary")
    print("=" * 60)
    print(f"Total masks scanned : {len(mask_files):,}")
    print(f"Total unique values : {len(sorted_classes)}")
    print(f"Min value found     : {min(sorted_classes) if sorted_classes else 'N/A'}")
    print(f"Max value found     : {max(sorted_classes) if sorted_classes else 'N/A'}")
    if 255 in sorted_classes:
        print("Note: Value 255 is detected (typically used as ignore_index / void).")
        print(f"Max valid class ID  : {max_class_id}")
        print(f"→ Recommended --num-classes : {max_class_id + 1} (or {len(valid_classes)})")
    else:
        print(f"→ Recommended --num-classes : {max_class_id + 1}")

    if len(sorted_classes) <= 50:
        print(f"\nAll Unique Values: {sorted_classes}")
    else:
        print(f"\nFirst 20 Unique Values: {sorted_classes[:20]} ... Last 5: {sorted_classes[-5:]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
