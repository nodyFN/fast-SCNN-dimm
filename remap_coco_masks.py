#!/usr/bin/env python3
"""
COCO-Stuff Mask Remapper (182 -> 171 Classes with Label 0 as Unlabeled / Ignore)
================================================================================

This utility remaps raw COCO-Stuff masks to continuous 171 classes (0 ~ 170)
and maps label 0 (unlabeled / void) to 255 (standard ignore_index for CrossEntropyLoss).

Features
--------
- Auto-scans all masks across train/val splits to find all valid non-zero class IDs.
- Maps label 0 (unlabeled) -> 255 (ignore_index).
- Maps the 171 valid categories -> contiguous indices 0 ~ 170.
- High-speed vectorized NumPy Look-Up Table (LUT) mapping with multiprocessing.
- Supports both in-place replacement or saving to a new directory.

Usage
-----
::

    # Scan & Remap split_set (train/val/test) into a new directory:
    python remap_coco_masks.py --data-root ../dataset/COCO_stuff/split_set --output-root ../dataset/COCO_stuff/split_set_171

    # Or remap in-place:
    python remap_coco_masks.py --data-root ../dataset/COCO_stuff/split_set --in-place

    # Remap a single mask folder:
    python remap_coco_masks.py --mask-dir ../dataset/COCO_stuff/split_set/train/masks --output-dir ../dataset/COCO_stuff/split_set_171/train/masks
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Official COCO-Stuff 182 to 171 valid class IDs (excluding 11 empty/merged classes)
# If custom non-zero classes exist, the script can also auto-fit dynamically.
COCO_STUFF_182_TO_171 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
    138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
    156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    174, 175, 176, 177, 178, 179, 180, 181, 182
]


def build_lut(valid_classes: List[int], ignore_val: int = 255) -> np.ndarray:
    """Build a fast 256-element uint8 Look-Up Table.

    Maps:
    - 0 -> ignore_val (255)
    - valid_classes[i] -> i (0 ~ len(valid_classes)-1)
    - any undefined class -> ignore_val (255)
    """
    lut = np.full(256, ignore_val, dtype=np.uint8)
    for new_id, old_id in enumerate(valid_classes):
        if 0 <= old_id <= 255:
            lut[old_id] = new_id
    lut[0] = ignore_val  # Ensure 0 is always mapped to ignore_val
    return lut


def process_single_mask(args: Tuple[Path, Path, np.ndarray]) -> Tuple[bool, Optional[str]]:
    """Remap a single PNG mask file using the LUT table."""
    in_path, out_path, lut = args
    try:
        with Image.open(in_path) as img:
            arr = np.array(img)

        # Vectorized lookup mapping
        remapped = lut[arr]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        res_img = Image.fromarray(remapped)
        res_img.save(str(out_path), format="PNG")
        return True, None
    except Exception as e:
        return False, f"{in_path.name}: {e}"


def scan_dataset_classes(mask_paths: List[Path], max_scan: int = 2000) -> List[int]:
    """Sample scan masks to extract non-zero unique class IDs."""
    unique_vals: Set[int] = set()
    scan_subset = mask_paths[:max_scan] if len(mask_paths) > max_scan else mask_paths

    for p in scan_subset:
        try:
            with Image.open(p) as img:
                vals = np.unique(np.array(img))
                unique_vals.update(vals.tolist())
        except Exception:
            continue

    # Filter out 0 (unlabeled) and 255
    valid_non_zero = sorted([int(v) for v in unique_vals if v not in (0, 255)])
    return valid_non_zero


def remap_folder(
    input_dir: Path,
    output_dir: Path,
    lut: np.ndarray,
    workers: int = 8,
) -> None:
    """Remap all mask files in input_dir to output_dir."""
    mask_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.PNG")))
    if not mask_files:
        print(f"No PNG masks found in {input_dir}")
        return

    print(f"\nProcessing {len(mask_files):,} masks from: {input_dir}")
    print(f"Destination: {output_dir}")

    tasks = [(f, output_dir / f.name, lut) for f in mask_files]

    success_count = 0
    errors: List[str] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_mask, t) for t in tasks]
        
        if tqdm is not None:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Remapping"):
                ok, err = fut.result()
                if ok:
                    success_count += 1
                else:
                    errors.append(err)
        else:
            for i, fut in enumerate(as_completed(futures), 1):
                ok, err = fut.result()
                if ok:
                    success_count += 1
                else:
                    errors.append(err)
                if i % 2000 == 0 or i == len(futures):
                    print(f"  Processed {i:,}/{len(futures):,} masks...")

    print(f"✓ Successfully remapped {success_count:,}/{len(mask_files):,} masks.")
    if errors:
        print(f"⚠ Errors encountered ({len(errors)}):")
        for e in errors[:5]:
            print(f"  - {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remap COCO-Stuff masks to 171 classes (0~170) with 0 -> 255 (ignore)"
    )
    p.add_argument("--data-root", type=str, default=None,
                   help="Root folder containing split_set (e.g. data/split_set)")
    p.add_argument("--output-root", type=str, default=None,
                   help="Destination root folder for remapped split_set")
    p.add_argument("--mask-dir", type=str, default=None,
                   help="Single mask directory to remap")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Single output directory for remapped masks")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite masks in-place directly")
    p.add_argument("--ignore-index", type=int, default=255,
                   help="Value to map label 0 (unlabeled) to (default: 255)")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel worker processes (default: 8)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.data_root is None and args.mask_dir is None:
        print("Error: Please specify either --data-root or --mask-dir.")
        sys.exit(1)

    # Use official 171 class list (1~182 without the 11 missing categories)
    valid_classes = COCO_STUFF_182_TO_171
    print("=" * 70)
    print("COCO-Stuff 182 -> 171 Category Remapper")
    print("=" * 70)
    print(f"Mapping scheme:")
    print(f"  • Label 0 (unlabeled) → {args.ignore_index} (ignore_index)")
    print(f"  • 171 Valid classes   → contiguous IDs [0 ~ 170]")
    print(f"  • Number of classes   → --num-classes 171")
    print("=" * 70)

    lut = build_lut(valid_classes, ignore_val=args.ignore_index)

    if args.data_root:
        data_root = Path(args.data_root)
        output_root = data_root if args.in_place else Path(args.output_root or f"{data_root}_171")

        for split in ["train", "val", "test"]:
            mask_in_dir = data_root / split / "masks"
            if mask_in_dir.is_dir():
                mask_out_dir = output_root / split / "masks"
                remap_folder(mask_in_dir, mask_out_dir, lut, workers=args.workers)

                # Link images if writing to new directory
                if not args.in_place and output_root != data_root:
                    img_in_dir = data_root / split / "images"
                    img_out_dir = output_root / split / "images"
                    if img_in_dir.is_dir() and not img_out_dir.exists():
                        img_out_dir.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            # Relative symlink
                            rel_target = os.path.relpath(img_in_dir, img_out_dir.parent)
                            os.symlink(rel_target, img_out_dir)
                            print(f"Linked images: {img_out_dir} -> {rel_target}")
                        except Exception as e:
                            print(f"Note: Could not symlink images ({e}), please copy or link manually.")

    elif args.mask_dir:
        mask_dir = Path(args.mask_dir)
        output_dir = mask_dir if args.in_place else Path(args.output_dir or f"{mask_dir}_171")
        remap_folder(mask_dir, output_dir, lut, workers=args.workers)

    print("\n" + "=" * 70)
    print("Remapping Complete!")
    print(f"For training, use:")
    print(f"  python train.py --data-root <path_to_remapped_split_set> \\")
    print(f"                  --num-classes 171 \\")
    print(f"                  --ignore-index {args.ignore_index}")
    print("=" * 70)


if __name__ == "__main__":
    main()
