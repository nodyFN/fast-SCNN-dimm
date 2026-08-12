#!/usr/bin/env python3
"""
Convert COCO-Stuff MATLAB annotation files (.mat) to standard PNG masks.

COCO-Stuff annotations contain a 2D integer matrix stored under the key 'S'.
This utility converts them into 8-bit grayscale PNG files preserving exact class IDs,
enabling fast I/O during training.

Usage
-----
::

    # Convert a directory of .mat files
    python convert_mat_to_png.py --mat-dir data/cocostuff/annotations --output-dir data/cocostuff/masks_png

    # Custom number of worker processes
    python convert_mat_to_png.py --mat-dir data/cocostuff/annotations --output-dir data/cocostuff/masks_png --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import scipy.io as sio
except ImportError:
    sio = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def convert_single_file(mat_path: Path, output_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """Convert a single .mat annotation file to a PNG mask.

    Returns
    -------
    (success, error_message, (min_class_id, max_class_id))
    """
    if sio is None:
        return False, "scipy is not installed (run `pip install scipy`)", None

    try:
        mat_data = sio.loadmat(str(mat_path))
        if "S" not in mat_data:
            return False, f"Key 'S' not found in {mat_path.name} (keys: {list(mat_data.keys())})", None

        mask = mat_data["S"]

        # Ensure 2D numpy array
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            return False, f"Unexpected mask shape {getattr(mask, 'shape', None)} in {mat_path.name}", None

        min_val = int(mask.min())
        max_val = int(mask.max())

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as 8-bit PNG (standard for <= 255 classes like COCO-Stuff 182)
        if max_val <= 255 and min_val >= 0:
            img = Image.fromarray(mask.astype(np.uint8))
        else:
            # 16-bit mode if class IDs exceed 255
            img = Image.fromarray(mask.astype(np.uint16))

        img.save(str(output_path), format="PNG")
        return True, None, (min_val, max_val)

    except Exception as e:
        return False, str(e), None


def _worker_wrapper(args: Tuple[Path, Path]) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    mat_path, output_path = args
    return convert_single_file(mat_path, output_path)


def convert_dataset(
    mat_dir: Path | str,
    output_dir: Path | str,
    workers: int = 8,
    recursive: bool = False,
) -> None:
    """Batch convert all .mat files in a directory to PNG masks."""
    if sio is None:
        print("Error: 'scipy' is required to read .mat files. Please install it with:\n  pip install scipy")
        sys.exit(1)

    mat_dir = Path(mat_dir)
    output_dir = Path(output_dir)

    if not mat_dir.exists():
        print(f"Error: Input directory does not exist: {mat_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect .mat files
    pattern = "**/*.mat" if recursive else "*.mat"
    mat_files = sorted(list(mat_dir.glob(pattern)))

    if not mat_files:
        print(f"No .mat files found in {mat_dir} (recursive={recursive})")
        return

    print("=" * 70)
    print(f"COCO-Stuff .mat to PNG Mask Converter")
    print(f"Input Directory  : {mat_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Total .mat files : {len(mat_files):,}")
    print(f"Worker Processes : {workers}")
    print("=" * 70)

    # Prepare file pairs
    tasks = []
    for f in mat_files:
        if recursive:
            rel = f.relative_to(mat_dir)
            out_file = output_dir / rel.with_suffix(".png")
        else:
            out_file = output_dir / f"{f.stem}.png"
        tasks.append((f, out_file))

    success_count = 0
    fail_count = 0
    global_min_class = float("inf")
    global_max_class = float("-inf")

    # Run multi-process conversion
    with ProcessPoolExecutor(max_workers=workers) as executor:
        if tqdm is not None:
            results_iter = tqdm(
                executor.map(_worker_wrapper, tasks),
                total=len(tasks),
                desc="Converting .mat to PNG",
                unit="file",
            )
            for success, err, stats in results_iter:
                if success:
                    success_count += 1
                    if stats:
                        global_min_class = min(global_min_class, stats[0])
                        global_max_class = max(global_max_class, stats[1])
                else:
                    fail_count += 1
                    if err:
                        tqdm.write(f"[WARN] Failed: {err}")
        else:
            for i, (success, err, stats) in enumerate(executor.map(_worker_wrapper, tasks), start=1):
                if success:
                    success_count += 1
                    if stats:
                        global_min_class = min(global_min_class, stats[0])
                        global_max_class = max(global_max_class, stats[1])
                else:
                    fail_count += 1
                    if err:
                        print(f"[WARN] Failed: {err}")
                if i % 1000 == 0 or i == len(tasks):
                    print(f"Progress: {i}/{len(tasks)} files processed...")

    print("=" * 70)
    print(f"Conversion Complete!")
    print(f"  Successfully converted : {success_count:,} files")
    print(f"  Failed                 : {fail_count:,} files")
    if success_count > 0 and global_min_class != float("inf"):
        print(f"  Class ID range in data : [{global_min_class} ~ {global_max_class}]")
    print(f"  Output directory       : {output_dir}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO-Stuff .mat annotations to PNG masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mat-dir", "-i", type=str, required=True,
        help="Path to folder containing .mat annotation files"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True,
        help="Path to output folder for PNG masks"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=os.cpu_count() or 4,
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recursively find .mat files in subdirectories"
    )
    args = parser.parse_args()

    convert_dataset(
        mat_dir=args.mat_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
