"""
Test suite for loss functions.

Verifies:
- Perfect prediction → near-minimum loss
- Foreground under-protection increases protection loss
- Background changes don't affect protection loss
- Empty foreground doesn't produce NaN
- Backward pass works
"""

import pytest
import torch

from utils.losses import DimmingLoss, ForegroundProtectionLoss


class TestForegroundProtectionLoss:
    """Test the foreground protection loss component."""

    def test_perfect_prediction(self):
        """prob=1 everywhere on foreground → loss ≈ 0."""
        loss_fn = ForegroundProtectionLoss()
        prob = torch.ones(2, 1, 32, 32)
        binary_mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(prob, binary_mask)
        assert loss.item() < 1e-5, f"Perfect prediction loss should be ~0, got {loss.item()}"

    def test_worst_prediction(self):
        """prob=0 everywhere on foreground → loss ≈ 1."""
        loss_fn = ForegroundProtectionLoss()
        prob = torch.zeros(2, 1, 32, 32)
        binary_mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(prob, binary_mask)
        assert abs(loss.item() - 1.0) < 1e-5, f"Worst prediction loss should be ~1, got {loss.item()}"

    def test_partial_prediction(self):
        """prob=0.5 on foreground → loss ≈ 0.5."""
        loss_fn = ForegroundProtectionLoss()
        prob = 0.5 * torch.ones(2, 1, 32, 32)
        binary_mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(prob, binary_mask)
        assert abs(loss.item() - 0.5) < 1e-5

    def test_under_protection_increases_loss(self):
        """Lowering foreground prediction should increase loss."""
        loss_fn = ForegroundProtectionLoss()
        binary_mask = torch.ones(1, 1, 16, 16)

        prob_high = 0.9 * torch.ones(1, 1, 16, 16)
        prob_low = 0.5 * torch.ones(1, 1, 16, 16)

        loss_high = loss_fn(prob_high, binary_mask)
        loss_low = loss_fn(prob_low, binary_mask)
        assert loss_low > loss_high, \
            f"Lower prediction should increase loss: {loss_low.item()} <= {loss_high.item()}"

    def test_background_does_not_affect_loss(self):
        """Changing background prediction should not change protection loss."""
        loss_fn = ForegroundProtectionLoss()
        binary_mask = torch.zeros(1, 1, 16, 16)
        binary_mask[:, :, 4:12, 4:12] = 1  # Small foreground

        # Same foreground prediction, different background
        prob1 = 0.8 * torch.ones(1, 1, 16, 16)
        prob2 = prob1.clone()
        prob2[:, :, 0:4, :] = 0.1  # Change background region

        loss1 = loss_fn(prob1, binary_mask)
        loss2 = loss_fn(prob2, binary_mask)
        assert abs(loss1.item() - loss2.item()) < 1e-6, \
            f"Background change should not affect loss: {loss1.item()} vs {loss2.item()}"

    def test_empty_foreground_no_nan(self):
        """No foreground pixels → loss should be 0, not NaN."""
        loss_fn = ForegroundProtectionLoss()
        prob = torch.rand(2, 1, 16, 16)
        binary_mask = torch.zeros(2, 1, 16, 16)
        loss = loss_fn(prob, binary_mask)
        assert not torch.isnan(loss), "Loss should not be NaN with empty foreground"
        assert loss.item() < 1e-5, "Loss should be ~0 with no foreground"


class TestDimmingLoss:
    """Test the composite DimmingLoss."""

    def test_output_keys(self):
        criterion = DimmingLoss()
        logits = torch.randn(2, 1, 32, 32)
        soft_target = torch.rand(2, 1, 32, 32)
        binary_mask = (soft_target > 0.5).float()
        losses = criterion(logits, soft_target, binary_mask)
        assert "total" in losses
        assert "bce" in losses
        assert "l1" in losses
        assert "protect" in losses

    def test_backward(self):
        """Verify gradients flow through the loss."""
        criterion = DimmingLoss()
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        soft_target = torch.rand(2, 1, 32, 32)
        binary_mask = (soft_target > 0.5).float()
        losses = criterion(logits, soft_target, binary_mask)
        losses["total"].backward()
        assert logits.grad is not None, "No gradient on logits"
        assert not torch.any(torch.isnan(logits.grad)), "NaN in gradients"

    def test_perfect_prediction_low_loss(self):
        """Near-perfect prediction should produce low total loss."""
        criterion = DimmingLoss()
        # Create a target
        soft_target = torch.zeros(1, 1, 32, 32)
        soft_target[:, :, 8:24, 8:24] = 1.0
        binary_mask = (soft_target > 0.5).float()

        # Create logits that would produce near-perfect sigmoid output
        # sigmoid(5) ≈ 0.9933
        logits = -5.0 * torch.ones(1, 1, 32, 32)
        logits[:, :, 8:24, 8:24] = 5.0

        losses = criterion(logits, soft_target, binary_mask)
        assert losses["total"].item() < 0.5, \
            f"Near-perfect prediction should have low loss, got {losses['total'].item()}"

    def test_zero_lambda_disables_component(self):
        """Setting lambda=0 should effectively disable that loss component."""
        logits = torch.randn(2, 1, 32, 32)
        soft_target = torch.rand(2, 1, 32, 32)
        binary_mask = (soft_target > 0.5).float()

        # Only BCE
        criterion = DimmingLoss(lambda_bce=1.0, lambda_l1=0.0, lambda_protect=0.0)
        losses = criterion(logits, soft_target, binary_mask)
        # total should equal bce
        assert abs(losses["total"].item() - losses["bce"].item()) < 1e-5

    def test_empty_foreground_no_nan(self):
        """All-background sample should not produce NaN."""
        criterion = DimmingLoss()
        logits = torch.randn(2, 1, 32, 32)
        soft_target = torch.zeros(2, 1, 32, 32)
        binary_mask = torch.zeros(2, 1, 32, 32)
        losses = criterion(logits, soft_target, binary_mask)
        for key, val in losses.items():
            assert not torch.isnan(val), f"NaN in {key}"

    def test_weighted_loss(self):
        """Verify that passing weight applies correctly and affects loss."""
        criterion = DimmingLoss()
        logits = torch.randn(2, 1, 32, 32)
        soft_target = torch.rand(2, 1, 32, 32)
        binary_mask = (soft_target > 0.5).float()

        # Create a weight mask that zeroes out the top half
        weight = torch.ones(2, 1, 32, 32)
        weight[:, :, :16, :] = 0.0

        losses_unweighted = criterion(logits, soft_target, binary_mask)
        losses_weighted = criterion(logits, soft_target, binary_mask, weight=weight)

        assert losses_weighted["total"].item() != losses_unweighted["total"].item()


class TestMulticlassLoss:
    """Test multiclass segmentation loss and criterion builder."""

    def test_multiclass_ce_loss(self):
        from utils.losses import MulticlassCrossEntropyLoss

        loss_fn = MulticlassCrossEntropyLoss(ignore_index=255)
        logits = torch.randn(2, 150, 32, 32, requires_grad=True)
        targets = torch.randint(0, 150, (2, 32, 32))
        targets[0, :5, :5] = 255  # test ignore_index

        res = loss_fn(logits, targets)
        assert "total" in res
        assert res["total"].item() > 0
        res["total"].backward()
        assert logits.grad is not None

    def test_build_criterion(self):
        from utils.losses import DimmingLoss, MulticlassCrossEntropyLoss, build_criterion

        crit_binary = build_criterion(num_classes=1)
        assert isinstance(crit_binary, DimmingLoss)

        crit_multi = build_criterion(num_classes=182)
        assert isinstance(crit_multi, MulticlassCrossEntropyLoss)


class TestDualHeadLoss:
    """Test the DualHeadLoss wrapper."""

    def test_edge_masking(self):
        from utils.losses import build_criterion
        criterion = build_criterion(
            num_classes=1,
            is_dual_head=True,
            coarse_edge_mask_kernel=15,
            coarse_target_dilation_kernel=15,
        )
        # Pred is a dict with fine_logits and coarse_logits
        pred = {
            "fine_logits": torch.randn(2, 1, 32, 32),
            "coarse_logits": torch.randn(2, 1, 32, 32),
        }
        soft_target = torch.rand(2, 1, 32, 32)
        binary_mask = (soft_target > 0.5).float()

        losses = criterion(pred, soft_target, binary_mask)
        assert "total" in losses
        assert "coarse_total" in losses
        assert "fine_total" in losses
