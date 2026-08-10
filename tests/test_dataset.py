"""
Test suite for dataset module.

Uses temporary synthetic images/masks to verify:
- {0, 1} mask format
- {0, 255} mask format
- Invalid mask handling
- Output dtypes and shapes
- Binary/soft mask synchronization
- Final shape exactly [1, 128, 224]
"""

import cv2
import numpy as np
import pytest
import tempfile
from pathlib import Path

from dataset import DimmingDataset

H, W = 128, 224


@pytest.fixture
def synthetic_dataset_01(tmp_path):
    """Create a synthetic dataset with {0, 1} masks."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    for i in range(5):
        # Create RGB image
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"sample_{i:03d}.jpg"), img)

        # Create {0, 1} mask
        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[50:150, 100:200] = 1
        cv2.imwrite(str(mask_dir / f"sample_{i:03d}.png"), mask)

    return tmp_path


@pytest.fixture
def synthetic_dataset_0255(tmp_path):
    """Create a synthetic dataset with {0, 255} masks."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    for i in range(5):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"sample_{i:03d}.jpg"), img)

        mask = np.zeros((200, 300), dtype=np.uint8)
        mask[50:150, 100:200] = 255
        cv2.imwrite(str(mask_dir / f"sample_{i:03d}.png"), mask)

    return tmp_path


@pytest.fixture
def synthetic_dataset_grayscale(tmp_path):
    """Create a synthetic dataset with grayscale masks (mixed values)."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "sample_000.jpg"), img)

    # Grayscale mask with values like 0, 128, 200, 255
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[50:100, 50:150] = 128
    mask[100:150, 50:150] = 200
    cv2.imwrite(str(mask_dir / "sample_000.png"), mask)

    return tmp_path


class TestMaskFormat01:
    """Test with {0, 1} binary masks."""

    def test_loads_successfully(self, synthetic_dataset_01):
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=False
        )
        assert len(ds) == 5

    def test_output_keys(self, synthetic_dataset_01):
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=False
        )
        sample = ds[0]
        assert "image" in sample
        assert "binary_mask" in sample
        assert "soft_mask" in sample
        assert "orig_size" in sample

    def test_output_shapes(self, synthetic_dataset_01):
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=False
        )
        sample = ds[0]
        assert sample["image"].shape == (3, H, W), \
            f"Image shape: {sample['image'].shape}"
        assert sample["binary_mask"].shape == (1, H, W), \
            f"Binary mask shape: {sample['binary_mask'].shape}"
        assert sample["soft_mask"].shape == (1, H, W), \
            f"Soft mask shape: {sample['soft_mask'].shape}"

    def test_output_dtypes(self, synthetic_dataset_01):
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=False
        )
        sample = ds[0]
        import torch
        assert sample["image"].dtype == torch.float32
        assert sample["binary_mask"].dtype == torch.float32
        assert sample["soft_mask"].dtype == torch.float32


class TestMaskFormat0255:
    """Test with {0, 255} binary masks."""

    def test_converts_correctly(self, synthetic_dataset_0255):
        ds = DimmingDataset(
            root=synthetic_dataset_0255, height=H, width=W, is_train=False
        )
        sample = ds[0]
        binary = sample["binary_mask"]
        # Should be converted to {0, 1}
        unique = binary.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique), \
            f"Expected {{0, 1}} mask, got unique values: {unique}"


class TestInvalidMask:
    """Test that grayscale masks raise error unless allow_threshold=True."""

    def test_raises_without_threshold(self, synthetic_dataset_grayscale):
        ds = DimmingDataset(
            root=synthetic_dataset_grayscale, height=H, width=W,
            is_train=False, allow_threshold=False,
        )
        with pytest.raises(ValueError, match="Unexpected mask values"):
            ds[0]

    def test_works_with_threshold(self, synthetic_dataset_grayscale):
        ds = DimmingDataset(
            root=synthetic_dataset_grayscale, height=H, width=W,
            is_train=False, allow_threshold=True,
        )
        sample = ds[0]
        binary = sample["binary_mask"]
        unique = binary.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique)


class TestBinarySoftSync:
    """Verify binary and soft masks are synchronized."""

    def test_foreground_implies_soft_one(self, synthetic_dataset_01):
        """Where binary_mask == 1, soft_mask should be 1.0."""
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=False
        )
        sample = ds[0]
        binary = sample["binary_mask"]
        soft = sample["soft_mask"]

        fg_mask = (binary == 1.0)
        if fg_mask.any():
            fg_soft_values = soft[fg_mask]
            assert (fg_soft_values == 1.0).all(), \
                f"Some foreground pixels have soft_mask != 1: min={fg_soft_values.min()}"

    def test_soft_range(self, synthetic_dataset_01):
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=False
        )
        sample = ds[0]
        soft = sample["soft_mask"]
        assert soft.min() >= 0.0
        assert soft.max() <= 1.0


class TestFinalShape:
    """Verify the exact output shape [1, 128, 224]."""

    def test_exact_shape_val(self, synthetic_dataset_01):
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=False
        )
        sample = ds[0]
        assert sample["binary_mask"].shape == (1, 128, 224)
        assert sample["soft_mask"].shape == (1, 128, 224)

    def test_exact_shape_train(self, synthetic_dataset_01):
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=True
        )
        sample = ds[0]
        assert sample["binary_mask"].shape == (1, 128, 224)
        assert sample["soft_mask"].shape == (1, 128, 224)

    def test_hw_not_swapped(self, synthetic_dataset_01):
        """H=128 < W=224 in spatial dimensions."""
        ds = DimmingDataset(
            root=synthetic_dataset_01, height=H, width=W, is_train=False
        )
        sample = ds[0]
        _, h, w = sample["image"].shape
        assert h == 128 and w == 224, f"Expected (128, 224), got ({h}, {w})"
        assert h < w, "H should be < W for landscape input"
