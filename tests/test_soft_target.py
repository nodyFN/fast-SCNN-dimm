"""
Test suite for soft target generation.

Uses small artificial masks to verify:
1. Output range [0, 1]
2. Original foreground all == 1.0
3. Dilation region all == 1.0
4. Transition extends only outward
5. Far background == 0.0
6. Protection value is monotonically non-increasing with distance
7. Cosine transition endpoints correct
8. Empty foreground doesn't crash
9. Full foreground doesn't crash
10. H/W not swapped
"""

import numpy as np
import pytest

from utils.soft_target import generate_soft_target


class TestSoftTargetRange:
    """Test output value range."""

    def test_range_01(self):
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[40:80, 80:160] = 1
        soft = generate_soft_target(mask)
        assert soft.min() >= 0.0, f"Min value {soft.min()} < 0"
        assert soft.max() <= 1.0, f"Max value {soft.max()} > 1"

    def test_dtype_float32(self):
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[40:80, 80:160] = 1
        soft = generate_soft_target(mask)
        assert soft.dtype == np.float32


class TestForegroundPreservation:
    """Original foreground pixels must be exactly 1.0."""

    def test_foreground_all_one(self):
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[40:80, 80:160] = 1
        soft = generate_soft_target(mask, protection_radius=2, transition_width=8)
        fg_values = soft[mask == 1]
        assert np.all(fg_values == 1.0), \
            f"Some foreground pixels are not 1.0: min={fg_values.min()}"

    def test_foreground_with_radius_0(self):
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[40:80, 80:160] = 1
        soft = generate_soft_target(mask, protection_radius=0, transition_width=8)
        assert np.all(soft[mask == 1] == 1.0)

    def test_foreground_with_large_radius(self):
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[60:70, 110:120] = 1
        soft = generate_soft_target(mask, protection_radius=10, transition_width=8)
        assert np.all(soft[mask == 1] == 1.0)


class TestDilationRegion:
    """Dilated protection region should be 1.0."""

    def test_dilation_region_is_one(self):
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[60:70, 100:120] = 1  # Small foreground

        # With protection_radius=2, dilated region should extend ~2px
        soft = generate_soft_target(mask, protection_radius=2, transition_width=8)

        # The center of the foreground should definitely be 1.0
        assert soft[65, 110] == 1.0
        # A pixel just outside the original foreground but within dilation
        # should also be 1.0 (e.g., 1 pixel outside)
        assert soft[59, 110] == 1.0  # 1px above original fg


class TestTransitionDirection:
    """Transition should extend outward from foreground, not inward."""

    def test_transition_only_outward(self):
        """Values at distance d should be <= values at distance d-1."""
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[50:80, 90:140] = 1  # Rectangle in center

        soft = generate_soft_target(mask, protection_radius=0, transition_width=20)

        # Walk outward from foreground boundary (top edge at row 50)
        # Rows 49, 48, 47, ... should have decreasing values
        prev_val = soft[50, 115]  # Inside foreground = 1.0
        for row in range(49, 30, -1):
            curr_val = soft[row, 115]
            assert curr_val <= prev_val + 1e-6, \
                f"Value at row {row} ({curr_val:.4f}) > row {row+1} ({prev_val:.4f})"
            prev_val = curr_val

    def test_monotonically_non_increasing(self):
        """Along a ray from foreground outward, values should not increase."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1

        soft = generate_soft_target(mask, protection_radius=2, transition_width=15)

        # Walk right from the foreground boundary
        prev_val = soft[30, 42]  # Inside dilation
        for col in range(43, 63):
            curr_val = soft[30, col]
            assert curr_val <= prev_val + 1e-6, \
                f"Non-decreasing at col {col}: {curr_val:.4f} > {prev_val:.4f}"
            prev_val = curr_val


class TestFarBackground:
    """Far background (distance >= transition_width) should be 0."""

    def test_far_background_is_zero(self):
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[60:70, 100:120] = 1

        soft = generate_soft_target(mask, protection_radius=2, transition_width=8)

        # Far corner — should be very far from foreground
        assert soft[0, 0] == 0.0, f"Far corner should be 0, got {soft[0, 0]}"
        assert soft[127, 223] == 0.0, f"Far corner should be 0, got {soft[127, 223]}"


class TestCosineEndpoints:
    """Verify cosine transition formula: M(0)=1, M(T)=0."""

    def test_cosine_at_boundary(self):
        """At the protected foreground boundary, value should be 1.0."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1

        soft = generate_soft_target(mask, protection_radius=0, transition_width=10)

        # Boundary pixel (just inside foreground)
        assert soft[20, 20] == 1.0
        # Just outside boundary should be < 1.0 but > 0
        assert 0.0 < soft[19, 20] < 1.0

    def test_cosine_beyond_transition(self):
        """Beyond transition_width, value should be 0.0."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30, 30] = 1  # Single pixel

        soft = generate_soft_target(mask, protection_radius=0, transition_width=5)

        # Far away pixel
        assert soft[0, 0] == 0.0


class TestEdgeCases:
    """Edge cases: empty, full, tiny masks."""

    def test_empty_foreground(self):
        """No foreground — should not crash, all zeros."""
        mask = np.zeros((128, 224), dtype=np.uint8)
        soft = generate_soft_target(mask)
        assert soft.shape == (128, 224)
        assert np.all(soft == 0.0)

    def test_full_foreground(self):
        """All foreground — should not crash, all ones."""
        mask = np.ones((128, 224), dtype=np.uint8)
        soft = generate_soft_target(mask)
        assert soft.shape == (128, 224)
        assert np.all(soft == 1.0)

    def test_single_pixel(self):
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[64, 112] = 1
        soft = generate_soft_target(mask, protection_radius=2, transition_width=8)
        assert soft[64, 112] == 1.0
        assert soft.min() >= 0.0
        assert soft.max() <= 1.0


class TestHWOrientation:
    """Verify H/W are not swapped."""

    def test_output_shape_matches_input(self):
        """Output shape should match input (H, W)."""
        mask = np.zeros((128, 224), dtype=np.uint8)
        mask[40:80, 80:160] = 1
        soft = generate_soft_target(mask)
        assert soft.shape == (128, 224), f"Expected (128, 224), got {soft.shape}"

    def test_asymmetric_mask(self):
        """Horizontal foreground should produce horizontal soft target."""
        mask = np.zeros((128, 224), dtype=np.uint8)
        # Wide horizontal bar
        mask[60:68, 20:200] = 1
        soft = generate_soft_target(mask, protection_radius=0, transition_width=10)

        # Check that the soft target is wider than tall
        nonzero_rows = np.any(soft > 0, axis=1)
        nonzero_cols = np.any(soft > 0, axis=0)
        h_extent = nonzero_rows.sum()
        w_extent = nonzero_cols.sum()
        assert w_extent > h_extent, \
            f"Wide bar should produce wider soft target, got h={h_extent}, w={w_extent}"


class TestModes:
    """Test different soft target modes."""

    def test_cosine_mode(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        soft = generate_soft_target(mask, mode="cosine")
        assert soft.shape == (64, 64)
        assert np.all(soft[mask == 1] == 1.0)

    def test_linear_mode(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        soft = generate_soft_target(mask, mode="linear")
        assert soft.shape == (64, 64)
        assert np.all(soft[mask == 1] == 1.0)

    def test_invalid_mode(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown soft target mode"):
            generate_soft_target(mask, mode="invalid")
