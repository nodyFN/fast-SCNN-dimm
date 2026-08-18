#!/usr/bin/env python3
"""
Utility script to convert grayscale images (e.g. soft masks) to binary masks.
Supports single files or entire folders.
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert grayscale masks to binary masks")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input grayscale image or directory of images",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save binary output masks",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Grayscale binarization threshold in range [0, 255] (default: 127)",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix to append to output file names (e.g. '_binary')",
    )
    return p.parse_args()


def process_image(img_path: Path, output_dir: Path, threshold: int, suffix: str) -> None:
    # Read as grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Failed to load image: {img_path}")
        return

    # Binarize: pixels >= threshold become 255, others become 0
    binary = (img >= threshold).astype(np.uint8) * 255

    # Determine output name
    out_name = f"{img_path.stem}{suffix}{img_path.suffix}"
    out_path = output_dir / out_name

    # Write output
    cv2.imwrite(str(out_path), binary)
    logger.info(f"Processed: {img_path.name} -> {out_path.name}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input paths
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    if input_path.is_file():
        process_image(input_path, output_dir, args.threshold, args.suffix)
    elif input_path.is_dir():
        # Find all common image formats
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
        img_paths = []
        for ext in extensions:
            img_paths.extend(input_path.glob(ext))

        # Sort for deterministic processing order
        img_paths.sort()

        if not img_paths:
            logger.warning(f"No images found in directory: {input_path}")
            return

        logger.info(f"Found {len(img_paths)} images in {input_path}. Starting processing...")
        for p in img_paths:
            process_image(p, output_dir, args.threshold, args.suffix)

    logger.info("Done!")


if __name__ == "__main__":
    main()
