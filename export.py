#!/usr/bin/env python3
"""
ONNX export for Fast-SCNN Dimming.

Exports the model to ONNX format and validates:
- Output shape
- Numerical comparison with PyTorch

Usage
-----
::

    python export.py --weights checkpoints/.../best_val_loss.pt

    # Include sigmoid in graph
    python export.py --weights checkpoints/.../best_val_loss.pt --include-sigmoid
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config
from models.fast_scnn_dimming import FastSCNNDimming, count_parameters
from utils.checkpoint import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FastSCNNDimmingWithSigmoid(nn.Module):
    """Wrapper that adds sigmoid to the model output for ONNX export."""

    def __init__(self, model: FastSCNNDimming) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Fast-SCNN Dimming to ONNX")
    p.add_argument("--weights", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--output", type=str, default=None, help="Output ONNX file")
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version [PROJECT DECISION]")
    p.add_argument("--include-sigmoid", action="store_true",
                   help="Include sigmoid in ONNX graph")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip-validation", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load model
    model = FastSCNNDimming(
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        dropout_p=0.0,  # Disable dropout for export
    ).to(device)

    ckpt = load_checkpoint(args.weights, model, map_location=device, weights_only=True)
    model.eval()
    logger.info(f"Loaded weights from: {args.weights}")

    total_params, _ = count_parameters(model)
    logger.info(f"Model parameters: {total_params:,}")

    # Wrap with sigmoid if requested
    if args.include_sigmoid:
        export_model = FastSCNNDimmingWithSigmoid(model)
        logger.info("Including sigmoid in ONNX graph")
    else:
        export_model = model
        logger.info("Exporting raw logits (no sigmoid)")

    export_model.eval()

    # Output path
    if args.output:
        onnx_path = Path(args.output)
    else:
        cfg.export_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_sigmoid" if args.include_sigmoid else "_logits"
        onnx_path = cfg.export_dir / f"fast_scnn_dimming{suffix}.onnx"

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input
    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    # Export
    logger.info(f"Exporting to: {onnx_path}")
    logger.info(f"  Input shape:  [1, 3, {args.height}, {args.width}]")
    logger.info(f"  Output shape: [1, 1, {args.height}, {args.width}]")
    logger.info(f"  Opset: {args.opset}")

    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info(f"ONNX export successful: {onnx_path}")

    # Validation
    if not args.skip_validation:
        _validate_onnx(export_model, onnx_path, dummy, device, args.include_sigmoid)


def _validate_onnx(
    model: nn.Module,
    onnx_path: Path,
    dummy: torch.Tensor,
    device: torch.device,
    include_sigmoid: bool,
) -> None:
    """Validate ONNX model against PyTorch."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnx / onnxruntime not installed. Skipping validation. "
            "Install with: pip install onnx onnxruntime"
        )
        return

    # 1. Check ONNX model is valid
    logger.info("\nValidating ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("  ONNX model check: ✓")

    # 2. Shape validation
    input_shape = onnx_model.graph.input[0].type.tensor_type.shape
    output_shape = onnx_model.graph.output[0].type.tensor_type.shape
    logger.info(f"  Input shape:  {[d.dim_value or d.dim_param for d in input_shape.dim]}")
    logger.info(f"  Output shape: {[d.dim_value or d.dim_param for d in output_shape.dim]}")

    # 3. Numerical comparison
    logger.info("  Running numerical comparison...")
    session = ort.InferenceSession(str(onnx_path))
    dummy_np = dummy.cpu().numpy()

    # PyTorch
    with torch.inference_mode():
        pt_out = model(dummy).cpu().numpy()

    # ONNX Runtime
    ort_out = session.run(None, {"input": dummy_np})[0]

    # Compare
    max_diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    logger.info(f"  Max absolute difference:  {max_diff:.8f}")
    logger.info(f"  Mean absolute difference: {mean_diff:.8f}")

    if max_diff < 1e-4:
        logger.info("  Numerical validation: ✓ (max diff < 1e-4)")
    elif max_diff < 1e-3:
        logger.info("  Numerical validation: ⚠ (max diff < 1e-3, acceptable)")
    else:
        logger.warning(f"  Numerical validation: ✗ (max diff = {max_diff})")


if __name__ == "__main__":
    main()
