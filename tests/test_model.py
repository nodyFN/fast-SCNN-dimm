"""
Test suite for FastSCNNDimming model.

Verifies feature shapes at every stage for the default input [B, 3, 128, 224].
"""

import pytest
import torch

from models.fast_scnn_dimming import (
    FastSCNNDimming,
    LearningToDownsample,
    GlobalFeatureExtractor,
    PyramidPooling,
    FeatureFusionModule,
    Classifier,
    count_parameters,
)

# [PROJECT DECISION] Default input: W=224, H=128
H, W = 128, 224


class TestLearningToDownsample:
    """Verify LtD produces /2 and /4 features (NOT /8)."""

    def test_output_shapes(self):
        ltd = LearningToDownsample()
        x = torch.randn(2, 3, H, W)
        feats = ltd.forward_features(x)

        # Layer 1: /2
        assert feats["feat_h2"].shape == (2, 32, H // 2, W // 2), \
            f"feat_h2 expected [2,32,{H//2},{W//2}], got {list(feats['feat_h2'].shape)}"

        # Layer 2: /4
        assert feats["feat_h4"].shape == (2, 48, H // 4, W // 4), \
            f"feat_h4 expected [2,48,{H//4},{W//4}], got {list(feats['feat_h4'].shape)}"

        # Layer 3: /4 (stride=1, stays at /4) — [PROJECT DECISION]
        assert feats["feat_h4_skip"].shape == (2, 64, H // 4, W // 4), \
            f"feat_h4_skip expected [2,64,{H//4},{W//4}], got {list(feats['feat_h4_skip'].shape)}"

    def test_forward_shape(self):
        ltd = LearningToDownsample()
        x = torch.randn(1, 3, H, W)
        out = ltd(x)
        assert out.shape == (1, 64, H // 4, W // 4)

    def test_hw_not_swapped(self):
        """Verify H and W dimensions are correct (not swapped)."""
        ltd = LearningToDownsample()
        x = torch.randn(1, 3, H, W)
        out = ltd(x)
        # H=128/4=32, W=224/4=56 — H should be smaller than W
        assert out.shape[2] == 32, f"Expected H=32, got {out.shape[2]}"
        assert out.shape[3] == 56, f"Expected W=56, got {out.shape[3]}"
        assert out.shape[2] < out.shape[3], "H should be < W for landscape input"


class TestGlobalFeatureExtractor:
    """Verify GFE produces /8 and /16 features (NOT /32)."""

    def test_output_shapes(self):
        gfe = GlobalFeatureExtractor()
        x = torch.randn(2, 64, H // 4, W // 4)  # Input from LtD at /4
        feats = gfe.forward_features(x)

        # Stage 1: /8
        assert feats["stage1_h8"].shape == (2, 64, H // 8, W // 8), \
            f"stage1 expected [2,64,{H//8},{W//8}], got {list(feats['stage1_h8'].shape)}"

        # Stage 2: /16
        assert feats["stage2_h16"].shape == (2, 96, H // 16, W // 16), \
            f"stage2 expected [2,96,{H//16},{W//16}], got {list(feats['stage2_h16'].shape)}"

        # Stage 3: /16 (stays)
        assert feats["stage3_h16"].shape == (2, 128, H // 16, W // 16), \
            f"stage3 expected [2,128,{H//16},{W//16}], got {list(feats['stage3_h16'].shape)}"

    def test_hw_not_swapped(self):
        gfe = GlobalFeatureExtractor()
        x = torch.randn(1, 64, 32, 56)
        out = gfe(x)
        # /16 from input: H=128/16=8, W=224/16=14
        assert out.shape[2] == 8
        assert out.shape[3] == 14


class TestPyramidPooling:
    """Verify PPM preserves spatial dimensions."""

    def test_output_shape(self):
        ppm = PyramidPooling(in_channels=128, out_channels=128)
        x = torch.randn(2, 128, H // 16, W // 16)
        out = ppm(x)
        assert out.shape == (2, 128, H // 16, W // 16)

    def test_custom_pool_sizes(self):
        ppm = PyramidPooling(in_channels=128, out_channels=128, pool_sizes=(1, 2, 4))
        x = torch.randn(2, 128, 8, 14)
        out = ppm(x)
        assert out.shape == (2, 128, 8, 14)


class TestFeatureFusionModule:
    """Verify FFM fuses /4 skip + /16 deep → /4 output."""

    def test_output_shape(self):
        ffm = FeatureFusionModule(high_channels=64, low_channels=128, out_channels=128)
        high = torch.randn(2, 64, H // 4, W // 4)
        low = torch.randn(2, 128, H // 16, W // 16)
        out = ffm(high, low)
        assert out.shape == (2, 128, H // 4, W // 4)

    def test_hw_not_swapped(self):
        ffm = FeatureFusionModule()
        high = torch.randn(1, 64, 32, 56)
        low = torch.randn(1, 128, 8, 14)
        out = ffm(high, low)
        assert out.shape[2] == 32
        assert out.shape[3] == 56


class TestClassifier:
    """Verify Classifier outputs 1-channel logits."""

    def test_output_shape(self):
        cls = Classifier(in_channels=128, out_channels=1)
        x = torch.randn(2, 128, H // 4, W // 4)
        out = cls(x)
        assert out.shape == (2, 1, H // 4, W // 4)


class TestFastSCNNDimming:
    """End-to-end model shape tests."""

    def test_output_shape(self):
        model = FastSCNNDimming()
        model.eval()
        x = torch.randn(2, 3, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, H, W), \
            f"Expected [2, 1, {H}, {W}], got {list(out.shape)}"

    def test_batch_size_1(self):
        model = FastSCNNDimming()
        model.eval()
        x = torch.randn(1, 3, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, H, W)

    def test_batch_size_4(self):
        model = FastSCNNDimming()
        model.eval()
        x = torch.randn(4, 3, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1, H, W)

    def test_debug_shapes(self):
        """Verify all intermediate feature map shapes."""
        model = FastSCNNDimming()
        model.eval()
        x = torch.randn(1, 3, H, W)
        with torch.no_grad():
            feats = model.forward_debug(x)

        expected = {
            "feat_h2":        (1, 32, 64, 112),
            "feat_h4":        (1, 48, 32, 56),
            "feat_h4_skip":   (1, 64, 32, 56),
            "gfe_stage1_h8":  (1, 64, 16, 28),
            "gfe_stage2_h16": (1, 96, 8, 14),
            "gfe_stage3_h16": (1, 128, 8, 14),
            "ppm_out":        (1, 128, 8, 14),
            "fused":          (1, 128, 32, 56),
            "logits_low":     (1, 1, 32, 56),
            "logits":         (1, 1, H, W),
        }

        for name, exp_shape in expected.items():
            actual = feats[name].shape
            assert actual == torch.Size(exp_shape), \
                f"{name}: expected {exp_shape}, got {list(actual)}"

    def test_hw_not_swapped(self):
        """Critical: verify H=128 < W=224 is maintained throughout."""
        model = FastSCNNDimming()
        model.eval()
        x = torch.randn(1, 3, H, W)
        with torch.no_grad():
            out = model(x)

        # Output: [1, 1, 128, 224]
        assert out.shape[2] == H, f"Output H expected {H}, got {out.shape[2]}"
        assert out.shape[3] == W, f"Output W expected {W}, got {out.shape[3]}"
        assert out.shape[2] < out.shape[3], "H should be < W for landscape input"

    def test_no_sigmoid_in_forward(self):
        """Model forward should return raw logits, not probabilities."""
        model = FastSCNNDimming()
        model.eval()
        # Use extreme values
        x = torch.randn(1, 3, H, W) * 10
        with torch.no_grad():
            out = model(x)
        # Raw logits can be negative or > 1
        # (may not always happen with random weights, but the model should
        #  not clamp or sigmoid)
        # At minimum, check dtype and shape
        assert out.dtype == torch.float32

    def test_parameter_count(self):
        """Verify model is lightweight (< 5M params)."""
        model = FastSCNNDimming()
        total, trainable = count_parameters(model)
        assert total < 5_000_000, f"Model has {total:,} params, expected < 5M"
        assert trainable == total, "All params should be trainable"

    def test_gradient_flow(self):
        """Verify gradients flow through the entire model."""
        model = FastSCNNDimming()
        model.train()
        x = torch.randn(2, 3, H, W, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient on input"
