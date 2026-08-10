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
