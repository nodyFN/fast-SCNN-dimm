#!/usr/bin/env python3
"""
Split paired images and masks into standard project dataset structure:
    <output_dir>/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/

Pairs are matched by file stem (e.g. `000000000009.jpg` <-> `000000000009.png`).

Usage
-----
::

    # Default 80% train, 10% val, 10% test (copy files)
    python split_dataset.py --images path/to/coco/images --masks path/to/coco/masks --output-dir coco_data

    # Use symlinks instead of copying (instant, zero extra disk space)
    python split_dataset.py --images path/to/coco/images --masks path/to/coco/masks --output-dir coco_data --symlink

    # Custom ratios
    python split_dataset.py --images path/to/coco/images --masks path/to/coco/masks --output-dir coco_data --train-ratio 0.85 --val-ratio 0.15 --test-ratio 0.0
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}


def find_and_pair_files(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """Match image files with mask files by stem."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    # Build mask lookup table (stem -> Path)
    mask_map: Dict[str, Path] = {}
    for p in masks_dir.iterdir():
        if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS:
            mask_map[p.stem] = p

    pairs: List[Tuple[Path, Path]] = []
    unmatched_images = 0

    for img_path in sorted(images_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = img_path.stem
            if stem in mask_map:
                pairs.append((img_path, mask_map[stem]))
            else:
                unmatched_images += 1

    if unmatched_images > 0:
        print(f"[WARN] {unmatched_images:,} images had no matching mask in {masks_dir}")

    return pairs


def transfer_file(src: Path, dst: Path, mode: str = "copy") -> None:
    """Transfer file using copy, symlink, or hardlink."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        # Create absolute symlink
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src.resolve(), dst)
    else:  # copy
        shutil.copy2(src, dst)


def _worker_copy(task: Tuple[Path, Path, str]) -> None:
    src, dst, mode = task
    transfer_file(src, dst, mode)


def split_and_organize(
    images_dir: Path | str,
    masks_dir: Path | str,
    output_dir: Path | str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    mode: str = "copy",
    workers: int = 8,
) -> None:
    """Split dataset and organize into train/val/test subdirectories."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0 (got train={train_ratio} + val={val_ratio} + test={test_ratio} = {total_ratio})"
        )

    print("=" * 70)
    print("Dataset Splitter & Organizer")
    print(f"Images Source    : {images_dir}")
    print(f"Masks Source     : {masks_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Split Ratios     : train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    print(f"Transfer Mode    : {mode}")
    print("=" * 70)

    # 1. Collect pairs
    print("Scanning and matching image/mask pairs...")
    pairs = find_and_pair_files(images_dir, masks_dir)
    total_samples = len(pairs)
    print(f"Total matched pairs found: {total_samples:,}")

    if total_samples == 0:
        print("Error: No matching pairs found between images and masks.")
        return

    # 2. Shuffle and split
    random.seed(seed)
    random.shuffle(pairs)

    n_train = int(round(total_samples * train_ratio))
    n_val = int(round(total_samples * val_ratio))
    # Remaining for test
    n_test = total_samples - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    print("\nDataset Split Breakdown:")
    for name, sp_pairs in splits.items():
        pct = (len(sp_pairs) / total_samples) * 100 if total_samples > 0 else 0
        print(f"  • {name:<5}: {len(sp_pairs):>7,} samples ({pct:>5.1f}%)")
    print("-" * 70)

    # 3. Build transfer tasks
    tasks = []
    for split_name, sp_pairs in splits.items():
        if not sp_pairs:
            continue
        dest_img_dir = output_dir / split_name / "images"
        dest_msk_dir = output_dir / split_name / "masks"
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_msk_dir.mkdir(parents=True, exist_ok=True)

        for img_src, mask_src in sp_pairs:
            tasks.append((img_src, dest_img_dir / img_src.name, mode))
            tasks.append((mask_src, dest_msk_dir / mask_src.name, mode))

    # 4. Execute transfers with thread pool
    print(f"Transferring {len(tasks):,} files ({mode})...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        if tqdm is not None:
            list(tqdm(executor.map(_worker_copy, tasks), total=len(tasks), desc="Processing files"))
        else:
            for i, _ in enumerate(executor.map(_worker_copy, tasks), start=1):
                if i % 2000 == 0 or i == len(tasks):
                    print(f"  Transferred {i}/{len(tasks)} files...")

    print("=" * 70)
    print("Dataset Split Complete!")
    print(f"Your dataset is ready at: {output_dir}")
    print("Structure:")
    print(f"  {output_dir}/")
    for name in ["train", "val", "test"]:
        if splits[name]:
            print(f"  ├── {name}/ ({len(splits[name]):,} pairs)")
            print(f"  │   ├── images/")
            print(f"  │   └── masks/")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split image and mask folders into train/val/test dataset layout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", "-i", type=str, required=True,
                        help="Path to source images folder")
    parser.add_argument("--masks", "-m", type=str, required=True,
                        help="Path to source masks folder")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="Output directory to create train/val/test folders")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Ratio for training set")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Ratio for validation set")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Ratio for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible shuffling")
    parser.add_argument("--symlink", action="store_true",
                        help="Use symlinks instead of copying files (faster, saves disk)")
    parser.add_argument("--hardlink", action="store_true",
                        help="Use hardlinks instead of copying files")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                        help="Number of concurrent workers for file transfer")
    args = parser.parse_args()

    mode = "copy"
    if args.symlink:
        mode = "symlink"
    elif args.hardlink:
        mode = "hardlink"

    split_and_organize(
        images_dir=args.images,
        masks_dir=args.masks,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        mode=mode,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
