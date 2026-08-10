#!/usr/bin/env python3
"""
Calculate MACs (Multiply-Accumulate Operations) and Parameter Counts
for Fast-SCNN Dimming at custom input resolutions.

Usage
-----
::

    # Default landscape resolution (128x224)
    python calculate_macs.py

    # Custom resolution
    python calculate_macs.py --height 128 --width 224

    # Output to CSV or Excel
    python calculate_macs.py --height 128 --width 224 --output macs_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import Config
from models.fast_scnn_dimming import FastSCNNDimming, count_parameters


class MACProfiler:
    """Hooks-based FLOPs/MACs and parameter profiler for PyTorch models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.records: List[Dict[str, Any]] = []
        self.hooks: List[Any] = []
        self._register_hooks(self.model, "")

    def _get_stage_name(self, layer_name: str) -> str:
        """Map layer name to architectural block name."""
        top_module = layer_name.split(".")[0] if "." in layer_name else layer_name
        stage_map = {
            "learning_to_downsample": "1. Learning to Downsample (/4)",
            "global_feature_extractor": "2. Global Feature Extractor (/16)",
            "pyramid_pooling": "3. Pyramid Pooling Module (PPM)",
            "feature_fusion": "4. Feature Fusion Module (FFM)",
            "classifier": "5. Classifier (Output Head)",
        }
        return stage_map.get(top_module, top_module)

    def _register_hooks(self, module: nn.Module, prefix: str) -> None:
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            # Register hooks on leaf compute layers or specific operational layers
            if len(list(child.children())) == 0 or isinstance(
                child,
                (
                    nn.Conv2d,
                    nn.Linear,
                    nn.BatchNorm2d,
                    nn.ReLU,
                    nn.MaxPool2d,
                    nn.AvgPool2d,
                    nn.AdaptiveAvgPool2d,
                ),
            ):
                h = child.register_forward_hook(self._make_hook(child_name))
                self.hooks.append(h)
            else:
                self._register_hooks(child, child_name)

    def _make_hook(self, name: str):
        def hook(module: nn.Module, input_args: Tuple[Any, ...], output: Any) -> None:
            in_shape = (
                tuple(input_args[0].shape)
                if input_args and len(input_args) > 0 and isinstance(input_args[0], torch.Tensor)
                else None
            )
            out_shape = (
                tuple(output.shape)
                if isinstance(output, torch.Tensor)
                else (
                    tuple(output[0].shape)
                    if isinstance(output, (list, tuple))
                    and len(output) > 0
                    and isinstance(output[0], torch.Tensor)
                    else None
                )
            )

            macs = 0
            params = sum(p.numel() for p in module.parameters())
            kernel_size = None
            stride = None

            if isinstance(module, nn.Conv2d) and out_shape is not None:
                kernel_size = module.kernel_size
                stride = module.stride
                groups = module.groups
                in_ch = module.in_channels
                out_ch = module.out_channels
                out_h, out_w = out_shape[-2:]
                k_h, k_w = kernel_size
                # MACs = Out_H * Out_W * Out_C * (In_C / groups) * K_H * K_W
                macs = out_h * out_w * out_ch * (in_ch // groups) * k_h * k_w
            elif isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                batch_size = out_shape[0] if out_shape else 1
                macs = in_features * out_features * batch_size

            stage = self._get_stage_name(name)

            self.records.append(
                {
                    "Stage": stage,
                    "Layer Name": name,
                    "Layer Type": module.__class__.__name__,
                    "Input Shape": str(in_shape) if in_shape else "",
                    "Output Shape": str(out_shape) if out_shape else "",
                    "Kernel Size": str(kernel_size) if kernel_size else "",
                    "Stride": str(stride) if stride else "",
                    "Parameters": params,
                    "MACs": macs,
                }
            )

        return hook

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []


def profile_fast_scnn_dimming(
    height: int = 128,
    width: int = 224,
    ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
    dropout_p: float = 0.1,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Profile FastSCNNDimming for parameters and MACs.

    Returns
    -------
    records : list of layer-by-layer profiling dicts
    total_params : int
    total_macs : int
    """
    device = torch.device("cpu")
    model = FastSCNNDimming(
        ppm_pool_sizes=ppm_pool_sizes,
        dropout_p=dropout_p,
    ).to(device)
    model.eval()

    total_params, _ = count_parameters(model)

    x = torch.randn(1, 3, height, width, device=device)
    profiler = MACProfiler(model)

    with torch.no_grad():
        _ = model(x)

    profiler.remove_hooks()

    total_macs = sum(r["MACs"] for r in profiler.records)
    return profiler.records, total_params, total_macs


def print_summary_table(
    records: List[Dict[str, Any]],
    height: int,
    width: int,
    total_params: int,
    total_macs: int,
) -> None:
    """Print a clean stage-by-stage summary table."""
    print("=" * 80)
    print(f"Fast-SCNN Dimming Complexity Profile — Input Shape: [1, 3, {height}, {width}]")
    print("=" * 80)

    # Subtotals by Stage
    stages: Dict[str, Dict[str, int]] = {}
    for r in records:
        st = r["Stage"]
        if st not in stages:
            stages[st] = {"MACs": 0, "Parameters": 0}
        stages[st]["MACs"] += r["MACs"]
        stages[st]["Parameters"] += r["Parameters"]

    header = f"{'Architectural Stage':<38} | {'Params':>12} | {'MACs (M)':>12} | {'% of MACs':>10}"
    print(header)
    print("-" * len(header))

    for st, data in stages.items():
        macs_m = data["MACs"] / 1e6
        pct = (data["MACs"] / max(total_macs, 1)) * 100
        print(f"{st:<38} | {data['Parameters']:>12,d} | {macs_m:>12.2f}M | {pct:>9.1f}%")

    print("=" * 80)
    print(f"{'Total Model Parameters':<38} : {total_params:,d} ({total_params/1e6:.3f} M)")
    print(f"{'Total Multiply-Accumulate (MACs)':<38} : {total_macs:,d} ({total_macs/1e6:.2f} M / {total_macs/1e9:.3f} GMACs)")
    print(f"{'Total FLOPs (~2 * MACs)':<38} : {2 * total_macs:,d} ({2 * total_macs/1e6:.2f} MFLOPs / {2 * total_macs/1e9:.3f} GFLOPs)")
    print("=" * 80)


def save_report(
    records: List[Dict[str, Any]],
    output_path: Path | str,
    total_params: int,
    total_macs: int,
) -> None:
    """Save detailed layer-by-layer records to CSV or Excel."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_row = {
        "Stage": "Total Summary",
        "Layer Name": "Total Model",
        "Layer Type": "FastSCNNDimming",
        "Input Shape": "",
        "Output Shape": "",
        "Kernel Size": "",
        "Stride": "",
        "Parameters": total_params,
        "MACs": total_macs,
    }
    all_records = list(records) + [summary_row]

    # Try using pandas if available
    try:
        import pandas as pd

        df = pd.DataFrame(all_records)
        if output_path.suffix.lower() in [".xlsx", ".xls"]:
            df.to_excel(output_path, sheet_name="fast_scnn_dimming", index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"\nDetailed report saved to: {output_path}")
    except ImportError:
        # Fallback to standard csv
        import csv

        keys = all_records[0].keys()
        csv_path = output_path.with_suffix(".csv") if output_path.suffix.lower() == ".xlsx" else output_path
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_records)
        print(f"\nDetailed CSV report saved to: {csv_path}")


def main() -> None:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Calculate Fast-SCNN Dimming MACs and Parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--height", "-H", type=int, default=cfg.train_height,
                        help=f"Target input height (default: {cfg.train_height})")
    parser.add_argument("--width", "-W", type=int, default=cfg.train_width,
                        help=f"Target input width (default: {cfg.train_width})")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional output file path (.csv or .xlsx)")
    args = parser.parse_args()

    records, total_params, total_macs = profile_fast_scnn_dimming(
        height=args.height,
        width=args.width,
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        dropout_p=cfg.dropout_p,
    )

    print_summary_table(records, args.height, args.width, total_params, total_macs)

    if args.output:
        save_report(records, args.output, total_params, total_macs)


if __name__ == "__main__":
    main()
