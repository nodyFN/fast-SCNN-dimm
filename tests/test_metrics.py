"""
Test suite for evaluation metrics.

Verifies metric correctness with known inputs.
"""

import torch
import pytest

from utils.metrics import MetricAccumulator


class TestBinaryMetrics:
    """Test binary segmentation metrics (IoU, Dice, Precision, Recall)."""

    def test_perfect_prediction(self):
        """Perfect prediction → all metrics = 1.0."""
        acc = MetricAccumulator()
        prob = torch.ones(1, 1, 16, 16)
        binary_mask = torch.ones(1, 1, 16, 16)
        soft_target = torch.ones(1, 1, 16, 16)
        acc.update(prob, soft_target, binary_mask)
        m = acc.compute()
        assert abs(m["fg_iou"] - 1.0) < 1e-4
        assert abs(m["dice"] - 1.0) < 1e-4
        assert abs(m["precision"] - 1.0) < 1e-4
        assert abs(m["recall"] - 1.0) < 1e-4

    def test_all_background_prediction(self):
        """Predict all background when GT is all foreground → recall = 0."""
        acc = MetricAccumulator()
        prob = torch.zeros(1, 1, 16, 16)
        binary_mask = torch.ones(1, 1, 16, 16)
        soft_target = torch.ones(1, 1, 16, 16)
        acc.update(prob, soft_target, binary_mask)
        m = acc.compute()
        assert m["recall"] < 1e-4
        assert m["fg_iou"] < 1e-4

    def test_all_foreground_prediction(self):
        """Predict all foreground when GT is all foreground → precision = 1."""
        acc = MetricAccumulator()
        prob = torch.ones(1, 1, 16, 16)
        binary_mask = torch.ones(1, 1, 16, 16)
        soft_target = torch.ones(1, 1, 16, 16)
        acc.update(prob, soft_target, binary_mask)
        m = acc.compute()
        assert abs(m["precision"] - 1.0) < 1e-4


class TestSoftMetrics:
    """Test soft mask quality metrics (MAE, MSE)."""

    def test_perfect_soft_prediction(self):
        acc = MetricAccumulator()
        target = torch.rand(1, 1, 16, 16)
        acc.update(target, target, (target > 0.5).float())
        m = acc.compute()
        assert m["soft_mae"] < 1e-6
        assert m["soft_mse"] < 1e-6

    def test_worst_soft_prediction(self):
        """Predict 0 when target is 1 → MAE = 1, MSE = 1."""
        acc = MetricAccumulator()
        prob = torch.zeros(1, 1, 16, 16)
        target = torch.ones(1, 1, 16, 16)
        binary = torch.ones(1, 1, 16, 16)
        acc.update(prob, target, binary)
        m = acc.compute()
        assert abs(m["soft_mae"] - 1.0) < 1e-4
        assert abs(m["soft_mse"] - 1.0) < 1e-4


class TestForegroundProtectionMetrics:
    """Test foreground protection metrics."""

    def test_perfect_protection(self):
        """prob=1 on all foreground → mean_protection=1, under_protection=0."""
        acc = MetricAccumulator()
        prob = torch.ones(1, 1, 16, 16)
        binary_mask = torch.ones(1, 1, 16, 16)
        soft_target = torch.ones(1, 1, 16, 16)
        acc.update(prob, soft_target, binary_mask)
        m = acc.compute()
        assert abs(m["fg_mean_protection"] - 1.0) < 1e-4
        assert m["fg_under_protection_error"] < 1e-4
        assert m["fg_under_protection_rate_09"] < 1e-4

    def test_low_protection(self):
        """prob=0.5 on foreground → mean_protection=0.5, under_rate_09 = 1.0."""
        acc = MetricAccumulator()
        prob = 0.5 * torch.ones(1, 1, 16, 16)
        binary_mask = torch.ones(1, 1, 16, 16)
        soft_target = torch.ones(1, 1, 16, 16)
        acc.update(prob, soft_target, binary_mask)
        m = acc.compute()
        assert abs(m["fg_mean_protection"] - 0.5) < 1e-4
        assert abs(m["fg_under_protection_error"] - 0.5) < 1e-4
        assert abs(m["fg_under_protection_rate_09"] - 1.0) < 1e-4

    def test_no_foreground(self):
        """No foreground → default values (protection=1, error=0)."""
        acc = MetricAccumulator()
        prob = torch.rand(1, 1, 16, 16)
        binary_mask = torch.zeros(1, 1, 16, 16)
        soft_target = torch.zeros(1, 1, 16, 16)
        acc.update(prob, soft_target, binary_mask)
        m = acc.compute()
        assert m["fg_mean_protection"] == 1.0
        assert m["fg_under_protection_error"] == 0.0


class TestFarBackgroundLeakage:
    """Test far-background leakage metric."""

    def test_no_leakage(self):
        """prob=0 where soft_target=0 → leakage=0."""
        acc = MetricAccumulator()
        prob = torch.zeros(1, 1, 16, 16)
        prob[:, :, 4:12, 4:12] = 1.0
        soft_target = torch.zeros(1, 1, 16, 16)
        soft_target[:, :, 4:12, 4:12] = 1.0
        binary_mask = (soft_target > 0.5).float()
        acc.update(prob, soft_target, binary_mask)
        m = acc.compute()
        assert m["far_bg_leakage"] < 1e-4

    def test_full_leakage(self):
        """prob=1 where soft_target=0 → leakage=1."""
        acc = MetricAccumulator()
        prob = torch.ones(1, 1, 16, 16)
        soft_target = torch.zeros(1, 1, 16, 16)
        binary_mask = torch.zeros(1, 1, 16, 16)
        acc.update(prob, soft_target, binary_mask)
        m = acc.compute()
        assert abs(m["far_bg_leakage"] - 1.0) < 1e-4


class TestMultipleBatches:
    """Test metric accumulation across multiple batches."""

    def test_accumulation(self):
        acc = MetricAccumulator()

        # Batch 1
        prob1 = torch.ones(2, 1, 16, 16)
        target1 = torch.ones(2, 1, 16, 16)
        binary1 = torch.ones(2, 1, 16, 16)
        acc.update(prob1, target1, binary1)

        # Batch 2
        prob2 = torch.zeros(2, 1, 16, 16)
        target2 = torch.zeros(2, 1, 16, 16)
        binary2 = torch.zeros(2, 1, 16, 16)
        acc.update(prob2, target2, binary2)

        m = acc.compute()
        # Should have accumulated across both batches
        assert m["soft_mae"] < 1e-4  # Both predictions match targets

    def test_reset(self):
        acc = MetricAccumulator()
        prob = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 16, 16)
        binary = (target > 0.5).float()
        acc.update(prob, target, binary)
        acc.reset()
        m = acc.compute()
        # After reset, counts should be 0
        assert m["soft_mae"] == 0.0
