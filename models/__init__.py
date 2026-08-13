"""Fast-SCNN Dimming model package."""

from typing import Tuple
import torch.nn as nn

from .fast_scnn_dimming import FastSCNNDimming, count_parameters
from .fast_scnn_dual_head import FastSCNNdualhead, FastSCNNDualHead


def build_model(
    model_name: str = "fast_scnn_dimming",
    num_classes: int = 1,
    ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
    dropout_p: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Factory function to build segmentation model by name.

    Parameters
    ----------
    model_name : str
        'fast_scnn_dimming' (default) | 'fast_scnn_dual_head'
    num_classes : int
        Number of output channels (1 for binary dimming mask).
    ppm_pool_sizes : tuple of int
        Pool sizes for PPM module.
    dropout_p : float
        Dropout probability.
    """
    name = str(model_name).lower().replace("-", "_")
    if name in ("fast_scnn_dual_head", "dual_head", "dualhead"):
        return FastSCNNdualhead(
            ppm_pool_sizes=ppm_pool_sizes,
            dropout_p=dropout_p,
            **kwargs,
        )
    elif name in ("fast_scnn_dimming", "dimming", "fast_scnn", "single_head"):
        return FastSCNNDimming(
            num_classes=num_classes,
            ppm_pool_sizes=ppm_pool_sizes,
            dropout_p=dropout_p,
        )
    else:
        raise ValueError(
            f"Unknown model_name: '{model_name}'. "
            f"Supported models: 'fast_scnn_dimming', 'fast_scnn_dual_head'"
        )
