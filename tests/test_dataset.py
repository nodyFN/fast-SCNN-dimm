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

    def test_custom_hw_resolution(self, synthetic_dataset_01):
        """Verify custom H/W resolution (e.g. 96x160) without hardcoded 128x224."""
        custom_h, custom_w = 96, 160
        ds_train = DimmingDataset(
            root=synthetic_dataset_01, height=custom_h, width=custom_w, is_train=True
        )
        sample = ds_train[0]
        assert sample["image"].shape == (3, custom_h, custom_w)
        assert sample["binary_mask"].shape == (1, custom_h, custom_w)
        assert sample["soft_mask"].shape == (1, custom_h, custom_w)

        ds_val = DimmingDataset(
            root=synthetic_dataset_01, height=custom_h, width=custom_w, is_train=False
        )
        sample_val = ds_val[0]
        assert sample_val["image"].shape == (3, custom_h, custom_w)
        assert sample_val["binary_mask"].shape == (1, custom_h, custom_w)
        assert sample_val["soft_mask"].shape == (1, custom_h, custom_w)

    def test_mask_nearest_neighbor_interpolation(self, tmp_path):
        """Verify that mask resize strictly uses nearest neighbor and produces ONLY {0.0, 1.0}."""
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        # Create image and fine checkerboard pattern mask to test interpolation
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "sample_diag.jpg"), img)

        # Diagonal line mask
        mask = np.zeros((200, 300), dtype=np.uint8)
        for r in range(200):
            mask[r, int(r * 1.5)] = 1
        cv2.imwrite(str(mask_dir / "sample_diag.png"), mask)

        # Test both train and val transforms
        for is_train in [True, False]:
            ds = DimmingDataset(
                root=tmp_path, height=128, width=224, is_train=is_train
            )
            sample = ds[0]
            binary_vals = sample["binary_mask"].unique().tolist()
            for v in binary_vals:
                assert v in [0.0, 1.0], f"Found non-binary value {v} in mask (is_train={is_train})"


class TestMatToPngConversion:
    """Test convert_mat_to_png utility."""

    def test_single_file_conversion(self, tmp_path):
        import scipy.io as sio
        from convert_mat_to_png import convert_single_file

        mat_file = tmp_path / "sample.mat"
        out_png = tmp_path / "sample.png"

        # Create dummy COCO-Stuff segmentation mask (182 classes)
        dummy_mask = np.random.randint(0, 182, (128, 224), dtype=np.uint8)
        sio.savemat(str(mat_file), {"S": dummy_mask})

        success, err, stats = convert_single_file(mat_file, out_png)
        assert success is True
        assert err is None
        assert out_png.exists()

        # Read back PNG and verify pixel values match exactly
        loaded_img = Image.open(str(out_png))
        loaded_arr = np.array(loaded_img)
        assert np.array_equal(loaded_arr, dummy_mask)
