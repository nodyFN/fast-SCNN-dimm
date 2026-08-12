"""
Dataset and DataLoader utilities for Foreground Protection / Dimming Soft Mask.

Data layout
-----------
::

    duts_data/
    ├── train/
    │   ├── images/   (*.jpg, *.jpeg, *.png)
    │   └── masks/    (*.png — single-channel, values 0/1 or 0/255)
    ├── val/
    │   └── ...
    └── test/
        └── ...

Image and mask are paired by matching file stem (e.g. ``ILSVRC2012_test_00000003``).

Each sample returns
-------------------
::

    {
        "image":       Tensor [3, H, W]   — ImageNet-normalized
        "binary_mask": Tensor [1, H, W]   — {0, 1}
        "soft_mask":   Tensor [1, H, W]   — [0, 1] cosine-feathered
        "orig_size":   Tuple[int, int]     — (orig_H, orig_W)
    }

IMPORTANT: augmentation ordering
---------------------------------
1. Load RGB + binary mask
2. Joint geometric augmentation (scale, crop, flip)
3. Resize to final resolution (H × W)
4. Binary mask uses nearest-neighbor interpolation
5. Generate soft target at final resolution
6. ImageNet normalize image

The soft target is generated AFTER resize, because protection_radius and
transition_width are defined in final-resolution pixel space.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.soft_target import generate_soft_target


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ===========================================================================
# Dataset
# ===========================================================================


class DimmingDataset(Dataset):
    """Foreground protection / dimming soft mask dataset.

    Parameters
    ----------
    root : Path or str
        Directory containing ``images/`` and ``masks/`` sub-directories.
    height, width : int
        Target spatial dimensions for model input.
        [PROJECT DECISION] default = 128 × 224.
    protection_radius : int
        Dilation radius for soft target.
    transition_width : int
        Cosine feather width for soft target.
    soft_target_mode : str
        "cosine" or "linear".
    is_train : bool
        If True, apply data augmentation.
    allow_threshold : bool
        If True, grayscale values > 127 → 1, ≤ 127 → 0.
    aug_scale_min, aug_scale_max : float
        Random scale range for training augmentation.
    aug_brightness_limit, aug_contrast_limit : float
        Brightness/contrast jitter range.
    aug_hflip_p : float
        Horizontal flip probability.
    """

    def __init__(
        self,
        root: Path | str,
        height: int = 128,
        width: int = 224,
        num_classes: int = 1,
        protection_radius: int = 2,
        transition_width: int = 8,
        soft_target_mode: str = "cosine",
        is_train: bool = True,
        allow_threshold: bool = False,
        aug_brightness_limit: float = 0.2,
        aug_contrast_limit: float = 0.2,
        aug_hflip_p: float = 0.5,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.protection_radius = protection_radius
        self.transition_width = transition_width
        self.soft_target_mode = soft_target_mode
        self.is_train = is_train
        self.allow_threshold = allow_threshold

        # Validate directories
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Discover and pair images + masks
        self.pairs = self._discover_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image/mask pairs found in {self.root}. "
                f"Ensure images/ and masks/ contain files with matching stems."
            )

        # Build augmentation pipeline
        self.transform = self._build_transform(
            is_train=is_train,
            aug_brightness_limit=aug_brightness_limit,
            contrast_limit=aug_contrast_limit,
            aug_hflip_p=aug_hflip_p,
        )

    def _discover_pairs(self) -> List[Tuple[Path, Path]]:
        """Find all (image_path, mask_path) pairs, sorted by stem."""
        image_map: Dict[str, Path] = {}
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                image_map[p.stem] = p

        mask_map: Dict[str, Path] = {}
        for p in sorted(self.mask_dir.iterdir()):
            if p.suffix.lower() == ".png":
                mask_map[p.stem] = p

        pairs: List[Tuple[Path, Path]] = []
        for stem in sorted(image_map.keys()):
            if stem not in mask_map:
                raise FileNotFoundError(
                    f"Mask not found for image '{image_map[stem].name}'. "
                    f"Expected a corresponding mask with stem '{stem}' in {self.mask_dir}"
                )
            pairs.append((image_map[stem], mask_map[stem]))
        return pairs

    def _build_transform(
        self,
        is_train: bool,
        aug_brightness_limit: float,
        contrast_limit: float,
        aug_hflip_p: float,
    ) -> A.Compose:
        """Build Albumentations transform pipeline.

        IMPORTANT ordering:
        1. Full-frame resize to target resolution (H x W)
           - Image uses cv2.INTER_LINEAR
           - Mask uses cv2.INTER_NEAREST (preserves binary values)
        2. Horizontal flip
        3. Color augmentations (brightness, contrast) — image only
        4. Normalize — image only
        5. ToTensor
        6. Soft target is generated from the FINAL binary mask after all transforms
        """
        if is_train:
            return A.Compose([
                # Full-frame resize: matches inference distribution
                A.Resize(
                    height=self.height,
                    width=self.width,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                ),
                # Horizontal flip
                A.HorizontalFlip(p=aug_hflip_p),
                # Color augmentations (image only, not mask)
                A.RandomBrightnessContrast(
                    brightness_limit=aug_brightness_limit,
                    contrast_limit=contrast_limit,
                    p=0.5,
                ),
                # Normalize
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(
                    height=self.height,
                    width=self.width,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ])

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask array based on num_classes."""
        if self.num_classes > 1:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"Failed to load mask: {mask_path}")
            return mask.astype(np.int64)
        return self._load_binary_mask(mask_path)

    def _load_binary_mask(self, mask_path: Path) -> np.ndarray:
        """Load and convert mask to binary {0, 1} uint8.

        Supports:
        - {0, 1}: use directly
        - {0, 255}: map 255 → 1
        - Other: raise ValueError (unless allow_threshold=True)
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")

        unique_vals = np.unique(mask)

        if set(unique_vals).issubset({0, 1}):
            return mask
        elif set(unique_vals).issubset({0, 255}):
            return (mask // 255).astype(np.uint8)
        elif self.allow_threshold:
            return (mask > 127).astype(np.uint8)
        else:
            raise ValueError(
                f"Unexpected mask values {unique_vals} in {mask_path}. "
                f"Expected {{0, 1}} or {{0, 255}}. "
                f"Set allow_threshold=True to threshold grayscale masks."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.pairs[idx]

        # Load image (RGB)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Load mask
        mask = self._load_mask(mask_path)

        # Apply augmentation (joint geometric + color + normalize + to tensor)
        # Albumentations uses INTER_NEAREST for masks by default in Resize
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]       # [3, H, W] float32
        mask_np = augmented["mask"]             # [H, W]

        if isinstance(mask_np, torch.Tensor):
            mask_np = mask_np.numpy()

        if self.num_classes > 1:
            # Multiclass segmentation mode (COCO-Stuff / ADE20K pretraining)
            mask_tensor = torch.from_numpy(mask_np.astype(np.int64)).unsqueeze(0)
            binary_mask_tensor = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)
            return {
                "image": image_tensor,
                "binary_mask": binary_mask_tensor,
                "soft_mask": mask_tensor,
                "orig_size": (orig_h, orig_w),
            }

        # Ensure binary mask is {0, 1} after augmentation
        mask_np = mask_np.astype(np.uint8)
        mask_np = (mask_np > 0).astype(np.uint8)  # Safety: re-binarize

        # Generate soft target at FINAL resolution
        # [PROJECT DECISION] soft target generated AFTER resize
        soft_mask_np = generate_soft_target(
            binary_mask=mask_np,
            protection_radius=self.protection_radius,
            transition_width=self.transition_width,
            mode=self.soft_target_mode,
        )

        # Convert to tensors: [1, H, W]
        binary_mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)
        soft_mask_tensor = torch.from_numpy(soft_mask_np).float().unsqueeze(0)

        return {
            "image": image_tensor,
            "binary_mask": binary_mask_tensor,
            "soft_mask": soft_mask_tensor,
            "orig_size": (orig_h, orig_w),
        }


# ===========================================================================
# DataLoader factory
# ===========================================================================


def build_dataloaders(
    data_root: Path | str,
    train_height: int = 128,
    train_width: int = 224,
    val_height: int = 128,
    val_width: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    num_classes: int = 1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    protection_radius: int = 2,
    transition_width: int = 8,
    soft_target_mode: str = "cosine",
    allow_threshold: bool = False,
    aug_brightness_limit: float = 0.2,
    aug_contrast_limit: float = 0.2,
    aug_hflip_p: float = 0.5,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders.

    Returns
    -------
    dict with keys "train", "val", and optionally "test".
    """
    data_root = Path(data_root)
    loaders: Dict[str, DataLoader] = {}

    for split, is_train in [("train", True), ("val", False), ("test", False)]:
        split_dir = data_root / split
        if not split_dir.exists():
            if split == "test":
                continue  # test split is optional
            raise FileNotFoundError(
                f"Expected {split} directory at {split_dir}"
            )

        if split == "test":
            mask_dir = split_dir / "masks"
            if not mask_dir.exists() or not any(mask_dir.glob("*.png")):
                continue  # test set has no annotations (e.g. ADE20K competition test set)

        h = train_height if is_train else val_height
        w = train_width if is_train else val_width

        try:
            dataset = DimmingDataset(
                root=split_dir,
                height=h,
                width=w,
                num_classes=num_classes,
                protection_radius=protection_radius,
                transition_width=transition_width,
                soft_target_mode=soft_target_mode,
                is_train=is_train,
                allow_threshold=allow_threshold,
                aug_brightness_limit=aug_brightness_limit,
                aug_contrast_limit=aug_contrast_limit,
                aug_hflip_p=aug_hflip_p,
            )
        except (FileNotFoundError, RuntimeError) as e:
            if split == "test":
                continue
            raise e

        use_persistent = persistent_workers and num_workers > 0

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=use_persistent,
            drop_last=is_train,
        )

    return loaders
