"""
Test suite for visualization utilities:
- Panel labeling
- Grayscale probability panel
- Training visualization generation
- Training curves plotting
"""

from pathlib import Path
import numpy as np
import pytest
import tempfile
import torch

from utils.visualization import (
    add_panel_label,
    plot_training_curves,
    save_side_by_side,
    save_training_visualization,
)


class TestVisualizationUtilities:
    """Test labeled panels and training visualization outputs."""

    def test_add_panel_label(self):
        panel = np.zeros((128, 224, 3), dtype=np.uint8)
        labeled = add_panel_label(panel, "Test Label", banner_height=22)
        assert labeled.shape == (128 + 22, 224, 3)

    def test_save_training_visualization(self, tmp_path):
        save_file = tmp_path / "val_vis.jpg"
        images = torch.randn(4, 3, 128, 224)
        binary_masks = torch.randint(0, 2, (4, 1, 128, 224)).float()
        soft_targets = torch.rand(4, 1, 128, 224)
        predictions = torch.rand(4, 1, 128, 224)

        save_training_visualization(
            save_file,
            images=images,
            binary_masks=binary_masks,
            soft_targets=soft_targets,
            predictions=predictions,
            num_samples=4,
        )
        assert save_file.exists()
        assert save_file.stat().st_size > 0

    def test_plot_training_curves(self, tmp_path):
        curve_file = tmp_path / "curves.png"
        history = {
            "epoch": [1, 2, 3],
            "train_loss": [0.8, 0.5, 0.3],
            "val_loss": [0.9, 0.6, 0.4],
            "fg_iou": [0.5, 0.7, 0.8],
            "miou": [0.6, 0.75, 0.85],
            "dice": [0.65, 0.80, 0.89],
            "fg_mean_protection": [0.8, 0.9, 0.95],
            "soft_mae": [0.1, 0.05, 0.03],
            "far_bg_leakage": [0.08, 0.04, 0.02],
        }
        plot_training_curves(history, curve_file)
        assert curve_file.exists()
        assert curve_file.stat().st_size > 0

    def test_multiclass_training_visualization(self, tmp_path):
        save_file = tmp_path / "val_vis_multiclass.jpg"
        images = torch.randn(4, 3, 128, 224)
        binary_masks = torch.randint(0, 2, (4, 1, 128, 224)).float()
        soft_targets = torch.randint(0, 183, (4, 1, 128, 224)).long()
        predictions = torch.randint(0, 183, (4, 1, 128, 224)).long()

        save_training_visualization(
            save_file,
            images=images,
            binary_masks=binary_masks,
            soft_targets=soft_targets,
            predictions=predictions,
            num_samples=4,
            num_classes=183,
        )
        assert save_file.exists()
        assert save_file.stat().st_size > 0
