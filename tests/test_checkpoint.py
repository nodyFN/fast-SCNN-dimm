"""
Test suite for checkpoint utilities and resume correctness.

Verifies:
- Checkpoint saving and loading
- Bookkeeping of best metrics (latest.pt holds the newest best values)
- State restoration (epoch, step, optimizer, scheduler, scaler)
"""

import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import pytest

from config import Config
from models.fast_scnn_dimming import FastSCNNDimming
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.scheduler import build_scheduler


class TestCheckpointResumeCorrectness:
    """Verify that latest.pt accurately preserves updated best metrics upon resume."""

    def test_latest_holds_updated_best_metrics(self):
        """
        Simulate training:
        Epoch 0: best_val_loss = 0.5
        Epoch 1: val_loss = 0.4 -> best_val_loss becomes 0.4
        Save latest.pt at epoch 1 -> reload -> assert best_val_loss == 0.4 (not 0.5)
        """
        cfg = Config()
        model = FastSCNNDimming()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler("poly", optimizer, total_iters=100, epochs=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "latest.pt"

            # Epoch 0: Initial best values
            best_val_loss = 0.5
            best_fg_protection = 0.85
            best_soft_mae = 0.08

            save_checkpoint(
                ckpt_path,
                epoch=0,
                global_step=10,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                best_fg_protection=best_fg_protection,
                best_soft_mae=best_soft_mae,
                config=asdict(cfg),
                seed=42,
            )

            # Epoch 1: A better epoch occurs
            epoch_1_val_loss = 0.40
            epoch_1_fg_prot = 0.95
            epoch_1_soft_mae = 0.03

            # Step 1: Update best values
            if epoch_1_val_loss < best_val_loss:
                best_val_loss = epoch_1_val_loss
            if epoch_1_fg_prot > best_fg_protection:
                best_fg_protection = epoch_1_fg_prot
            if epoch_1_soft_mae < best_soft_mae:
                best_soft_mae = epoch_1_soft_mae

            # Step 2: Save latest.pt with updated values
            save_checkpoint(
                ckpt_path,
                epoch=1,
                global_step=20,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                best_fg_protection=best_fg_protection,
                best_soft_mae=best_soft_mae,
                config=asdict(cfg),
                seed=42,
            )

            # Step 3: Load latest.pt into a new model and verify
            new_model = FastSCNNDimming()
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            new_scheduler = build_scheduler("poly", new_optimizer, total_iters=100, epochs=10)

            loaded_ckpt = load_checkpoint(
                ckpt_path,
                new_model,
                optimizer=new_optimizer,
                scheduler=new_scheduler,
            )

            assert loaded_ckpt["epoch"] == 1
            assert loaded_ckpt["global_step"] == 20
            assert abs(loaded_ckpt["best_val_loss"] - 0.40) < 1e-6, \
                f"Expected best_val_loss=0.40, got {loaded_ckpt['best_val_loss']}"
            assert abs(loaded_ckpt["best_fg_protection"] - 0.95) < 1e-6
            assert abs(loaded_ckpt["best_soft_mae"] - 0.03) < 1e-6

    def test_transfer_learning_mismatched_classifier(self):
        """Verify transfer learning across different class counts (COCO 182 -> ADE 150 -> Binary 1)."""
        from utils.checkpoint import load_pretrained_weights

        with tempfile.TemporaryDirectory() as tmp_dir:
            coco_ckpt_path = Path(tmp_dir) / "coco_backbone.pt"

            # 1. Simulate training on COCO-Stuff (num_classes=182)
            coco_model = FastSCNNDimming(num_classes=182)
            save_checkpoint(
                coco_ckpt_path,
                epoch=50,
                global_step=1000,
                model=coco_model,
                optimizer=torch.optim.AdamW(coco_model.parameters()),
                scheduler=None,
            )

            # 2. Transfer to ADE20K (num_classes=150)
            ade_model = FastSCNNDimming(num_classes=150)
            load_pretrained_weights(coco_ckpt_path, ade_model)

            # Verify backbone weights match
            for p_coco, p_ade in zip(
                coco_model.learning_to_downsample.parameters(),
                ade_model.learning_to_downsample.parameters(),
            ):
                assert torch.equal(p_coco, p_ade), "Backbone weights should match after transfer"

            # Verify classifier output works properly with 150 classes
            x = torch.randn(1, 3, 128, 224)
            with torch.no_grad():
                out_ade = ade_model(x)
            assert out_ade.shape == (1, 150, 128, 224)

            # 3. Transfer from ADE20K to Binary Dimming (num_classes=1)
            ade_ckpt_path = Path(tmp_dir) / "ade_backbone.pt"
            save_checkpoint(
                ade_ckpt_path,
                epoch=80,
                global_step=2000,
                model=ade_model,
                optimizer=torch.optim.AdamW(ade_model.parameters()),
                scheduler=None,
            )

            dimm_model = FastSCNNDimming(num_classes=1)
            load_pretrained_weights(ade_ckpt_path, dimm_model)

            # Verify backbone weights transferred to dimm_model
            for p_ade, p_dimm in zip(
                ade_model.learning_to_downsample.parameters(),
                dimm_model.learning_to_downsample.parameters(),
            ):
                assert torch.equal(p_ade, p_dimm)

            with torch.no_grad():
                out_dimm = dimm_model(x)
            assert out_dimm.shape == (1, 1, 128, 224)
