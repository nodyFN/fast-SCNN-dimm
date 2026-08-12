"""
Fast-SCNN Dimming: Foreground Protection / Dimming Soft Mask Prediction
========================================================================

Modified Fast-SCNN architecture for TV SoC foreground protection mask prediction.

Architecture overview
---------------------
::

    Input [B, 3, 128, 224]
          │
          ▼
    Learning to Downsample
          │
          ├──── shallow skip [B, 64, 32, 56] ──────────┐
          │                                             │
          ▼                                             │
    Global Feature Extractor                            │
          │                                             │
          ▼                                             │
    deep [B, 128, 8, 14]                                │
          │                                             │
          ▼                                             │
    Pyramid Pooling Module                              │
          │                                             │
          ▼                                             │
    Feature Fusion Module ◄─────────────────────────────┘
          │
          ▼
    [B, 128, 32, 56]
          │
          ▼
    Classifier
          │
          ▼
    [B, 1, 32, 56]
          │
          ▼
    Bilinear Upsample
          │
          ▼
    raw logits [B, 1, 128, 224]

Resolution changes from original Fast-SCNN
-------------------------------------------
[PROJECT DECISION] LtD output stride: /8 → /4
    Layer 3 stride changed from 2 to 1.
    Provides higher-resolution skip features for better mask detail.

[PROJECT DECISION] GFE deep output stride: /32 → /16
    Stage 2 stride=2 ends at /16.  Stage 3 stays /16.
    Larger deep feature maps for better spatial detail at cost of compute.

[PROJECT DECISION] Single-channel output (1 class logit, not 2).
    Binary foreground protection — use BCEWithLogitsLoss.

[PROJECT DECISION] Input resolution: W=224 × H=128 (landscape TV).

Paper settings retained
-----------------------
- Activation: ReLU (NOT ReLU6)
- BatchNorm after every conv (except where noted)
- Linear bottleneck: no activation after final 1×1 projection
- DSConv: no ReLU between depthwise and pointwise
- PPM pool_sizes default (1, 2, 3, 6)
- Kaiming initialization for conv weights, BN weight=1 / bias=0
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Building blocks
# ===========================================================================


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU (optional).

    When conv is followed by BN, bias is disabled by default.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        relu: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution.

    Structure (per paper):
        3×3 Depthwise Conv → BN → 1×1 Pointwise Conv → BN → ReLU

    No ReLU between depthwise and pointwise as specified by the paper.
    The final ReLU can be disabled via ``relu=False`` for modules that
    require a linear output before addition (e.g. FFM branches).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        relu: bool = True,
    ) -> None:
        super().__init__()
        # Depthwise
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        # Pointwise
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_dw(self.depthwise(x))
        x = self.bn_pw(self.pointwise(x))
        if self.relu is not None:
            x = self.relu(x)
        return x


class LinearBottleneck(nn.Module):
    """MobileNetV2-style Inverted Residual Linear Bottleneck.

    Structure:
        1×1 expand (c_in → t*c_in) → BN → ReLU
        3×3 depthwise (t*c_in, stride=s) → BN → ReLU
        1×1 project (t*c_in → c_out) → BN → NO activation

    Residual connection only when stride == 1 AND c_in == c_out.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 6,
        stride: int = 1,
    ) -> None:
        super().__init__()
        mid_channels = in_channels * expansion
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # Linear projection – no activation
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


# ===========================================================================
# Major modules
# ===========================================================================


class LearningToDownsample(nn.Module):
    """Learning to Downsample — three layers with total stride /4.

    [PROJECT DECISION] Changed from original /8 to /4:
        Layer 3 stride changed from 2 to 1.

    Layer 1: standard 3×3 Conv (stride 2)  3  → 32  → /2
    Layer 2: 3×3 DSConv (stride 2)         32 → 48  → /4
    Layer 3: 3×3 DSConv (stride 1)         48 → 64  → /4  (stays)

    For input [B, 3, 128, 224]:
        feat_h2      = [B, 32, 64, 112]
        feat_h4      = [B, 48, 32, 56]
        feat_h4_skip = [B, 64, 32, 56]   ← skip to FFM
    """

    def __init__(self) -> None:
        super().__init__()
        # Layer 1: standard conv (3 input channels make DSConv inefficient)
        self.conv = ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1)
        # Layer 2: DSConv stride=2
        self.dsconv1 = DepthwiseSeparableConv(32, 48, stride=2)
        # [PROJECT DECISION] Layer 3: DSConv stride=1 (was stride=2 in original)
        self.dsconv2 = DepthwiseSeparableConv(48, 64, stride=1)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward and return intermediate outputs at each layer."""
        feat_h2 = self.conv(x)           # [B, 32, H/2, W/2]
        feat_h4 = self.dsconv1(feat_h2)  # [B, 48, H/4, W/4]
        feat_h4_skip = self.dsconv2(feat_h4)  # [B, 64, H/4, W/4]
        return {
            "feat_h2": feat_h2,
            "feat_h4": feat_h4,
            "feat_h4_skip": feat_h4_skip,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global Feature Extractor using MobileNetV2 bottlenecks.

    [PROJECT DECISION] Deep output stride changed from /32 to /16:
        Stage 1 goes from /4 to /8.
        Stage 2 goes from /8 to /16.
        Stage 3 stays at /16.

    Three stages:
        Stage 1:  64 → 64,  t=6, n=3, s=2   (/4 → /8)
        Stage 2:  64 → 96,  t=6, n=3, s=2   (/8 → /16)
        Stage 3:  96 → 128, t=6, n=3, s=1   (stays /16)

    For input [B, 64, 32, 56]:
        after stage 1: [B, 64,  16, 28]
        after stage 2: [B, 96,   8, 14]
        after stage 3: [B, 128,  8, 14]   ← deep output
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            *self._make_stage(64, 64, expansion=6, num_blocks=3, stride=2)
        )
        self.stage2 = nn.Sequential(
            *self._make_stage(64, 96, expansion=6, num_blocks=3, stride=2)
        )
        self.stage3 = nn.Sequential(
            *self._make_stage(96, 128, expansion=6, num_blocks=3, stride=1)
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        expansion: int,
        num_blocks: int,
        stride: int,
    ) -> List[LinearBottleneck]:
        layers: List[LinearBottleneck] = []
        # First block uses the given stride and changes channels
        layers.append(
            LinearBottleneck(in_channels, out_channels, expansion, stride)
        )
        # Remaining blocks: stride=1, channels stay at out_channels
        for _ in range(1, num_blocks):
            layers.append(
                LinearBottleneck(out_channels, out_channels, expansion, 1)
            )
        return layers

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward and return intermediate outputs at each stage."""
        s1 = self.stage1(x)   # /8
        s2 = self.stage2(s1)  # /16
        s3 = self.stage3(s2)  # /16
        return {
            "stage1_h8": s1,
            "stage2_h16": s2,
            "stage3_h16": s3,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class PyramidPooling(nn.Module):
    """Pyramid Pooling Module (PPM).

    [PROJECT DECISION] The paper references PSPNet-style PPM but does not
    specify pool_sizes or branch channels for Fast-SCNN.

    Default: pool_sizes=(1, 2, 3, 6), branch_channels=32, out=128.

    WARNING: When batch_size=1, the 1×1 pooling branch produces a single
    spatial value per channel, which can cause BatchNorm instability during
    training.  Use batch_size > 1 for training, or freeze BN / switch to
    GroupNorm if needed.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
        branch_channels: int = 32,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for ps in pool_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        # Fusion: concat(original + branches) → 1×1 conv
        concat_channels = in_channels + branch_channels * len(pool_sizes)
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        branch_outs = [x]
        for branch in self.branches:
            b = branch(x)
            b = F.interpolate(
                b, size=input_size, mode="bilinear", align_corners=False
            )
            branch_outs.append(b)
        out = torch.cat(branch_outs, dim=1)
        return self.fusion(out)


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module (FFM).

    Fuses high-resolution skip features from LtD (64-ch, /4) with
    low-resolution global features from GFE+PPM (128-ch, /16).

    Low-res branch:
        Bilinear upsample to high-res spatial size
        → 3×3 DW conv (dilation=4, padding=4)
        → BN → ReLU
        → 1×1 PW conv → BN (no activation before add)

    High-res branch:
        1×1 Conv → BN (no activation before add)

    Fusion: element-wise add → ReLU

    ``align_corners=False`` is used for bilinear interpolation for stability
    across arbitrary input sizes and consistency between PyTorch / ONNX Runtime.
    """

    def __init__(
        self,
        high_channels: int = 64,
        low_channels: int = 128,
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        # Low-resolution branch
        self.low_dw = nn.Conv2d(
            low_channels,
            low_channels,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            groups=low_channels,
            bias=False,
        )
        self.low_bn_dw = nn.BatchNorm2d(low_channels)
        self.low_relu = nn.ReLU(inplace=True)
        self.low_pw = nn.Conv2d(low_channels, out_channels, 1, bias=False)
        self.low_bn_pw = nn.BatchNorm2d(out_channels)

        # High-resolution branch
        self.high_proj = nn.Conv2d(high_channels, out_channels, 1, bias=False)
        self.high_bn = nn.BatchNorm2d(out_channels)

        # Post-fusion activation
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        high_res: torch.Tensor,
        low_res: torch.Tensor,
    ) -> torch.Tensor:
        # Use actual spatial size of high-res feature for alignment
        target_size = high_res.shape[-2:]

        # Low-resolution branch: upsample first, then DW→BN→ReLU→PW→BN
        low = F.interpolate(
            low_res, size=target_size, mode="bilinear", align_corners=False
        )
        low = self.low_relu(self.low_bn_dw(self.low_dw(low)))
        low = self.low_bn_pw(self.low_pw(low))  # no activation

        # High-resolution branch: 1×1→BN (no activation)
        high = self.high_bn(self.high_proj(high_res))

        # Element-wise addition → ReLU
        return self.relu(high + low)


class Classifier(nn.Module):
    """Classifier head.

    Structure:
        DSConv 128→128 (stride 1)
        DSConv 128→128 (stride 1)
        Dropout  [PROJECT DECISION: p=0.1]
        1×1 Conv 128→num_classes (default 1 for binary dimming, >1 for multiclass pretraining)

    Output is raw logits — no sigmoid or softmax applied.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 1,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.dsconv1 = DepthwiseSeparableConv(in_channels, in_channels, stride=1)
        self.dsconv2 = DepthwiseSeparableConv(in_channels, in_channels, stride=1)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dropout(x)
        return self.conv(x)


# ===========================================================================
# Full model
# ===========================================================================


class FastSCNNDimming(nn.Module):
    """Fast-SCNN Dimming: Foreground Protection Soft Mask Prediction.

    Parameters
    ----------
    num_classes : int
        Number of output channels / classes.
        Default = 1 (binary foreground dimming mask).
        Set > 1 for multiclass semantic segmentation pretraining (e.g. COCO-Stuff, ADE20K).
    ppm_pool_sizes : tuple of int
        Pool sizes for the Pyramid Pooling Module.
        [PROJECT DECISION] default = (1, 2, 3, 6).
    dropout_p : float
        Dropout probability in the Classifier.
        [PROJECT DECISION] default = 0.1.

    Output
    ------
    raw logits : Tensor [B, num_classes, H, W]
        No activation applied.
    """

    def __init__(
        self,
        num_classes: int = 1,
        ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Main backbone
        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.ppm = PyramidPooling(
            in_channels=128,
            out_channels=128,
            pool_sizes=ppm_pool_sizes,
        )
        self.ffm = FeatureFusionModule(
            high_channels=64, low_channels=128, out_channels=128
        )
        self.classifier = Classifier(
            in_channels=128, out_channels=num_classes, dropout_p=dropout_p
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for conv weights; BN weight=1, bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits [B, 1, H, W].

        No sigmoid is applied — caller is responsible for activation.
        """
        input_size = x.shape[-2:]  # (H, W)

        # Learning to Downsample — produces skip feature at /4
        ltd_out = self.learning_to_downsample(x)  # [B, 64, H/4, W/4]

        # Global Feature Extractor — /16
        gfe_out = self.global_feature_extractor(ltd_out)  # [B, 128, H/16, W/16]

        # Pyramid Pooling — still /16, 128-ch
        ppm_out = self.ppm(gfe_out)  # [B, 128, H/16, W/16]

        # Feature Fusion — combines skip (/4) + global (/16) → /4
        fused = self.ffm(high_res=ltd_out, low_res=ppm_out)  # [B, 128, H/4, W/4]

        # Classifier → logits at /4, then upsample to input size
        logits = self.classifier(fused)  # [B, 1, H/4, W/4]
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )

        return logits  # [B, 1, H, W]

    def forward_debug(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning all intermediate feature maps for debugging.

        Useful for shape verification tests.
        """
        input_size = x.shape[-2:]

        # LtD features
        ltd_feats = self.learning_to_downsample.forward_features(x)

        # GFE features (from skip output)
        gfe_feats = self.global_feature_extractor.forward_features(
            ltd_feats["feat_h4_skip"]
        )

        # PPM
        ppm_out = self.ppm(gfe_feats["stage3_h16"])

        # FFM
        fused = self.ffm(
            high_res=ltd_feats["feat_h4_skip"],
            low_res=ppm_out,
        )

        # Classifier
        logits_low = self.classifier(fused)
        logits = F.interpolate(
            logits_low, size=input_size, mode="bilinear", align_corners=False
        )

        return {
            # LtD
            "feat_h2": ltd_feats["feat_h2"],
            "feat_h4": ltd_feats["feat_h4"],
            "feat_h4_skip": ltd_feats["feat_h4_skip"],
            # GFE
            "gfe_stage1_h8": gfe_feats["stage1_h8"],
            "gfe_stage2_h16": gfe_feats["stage2_h16"],
            "gfe_stage3_h16": gfe_feats["stage3_h16"],
            # PPM
            "ppm_out": ppm_out,
            # FFM
            "fused": fused,
            # Classifier
            "logits_low": logits_low,
            "logits": logits,
        }


# ===========================================================================
# Utilities
# ===========================================================================


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _benchmark(
    model: nn.Module,
    device: torch.device,
    height: int,
    width: int,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
) -> None:
    """Run FPS / latency benchmark."""
    model.eval()
    dummy = torch.randn(batch_size, 3, height, width, device=device)

    use_cuda = device.type == "cuda"

    # Warm-up
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize()

    # Timed iterations
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_images = batch_size * iterations
    avg_latency_ms = (total_time / iterations) * 1000
    fps = total_images / total_time

    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"  Device           : {device}")
    print(f"  Input resolution : {height}×{width}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Warm-up iters    : {warmup}")
    print(f"  Timed iters      : {iterations}")
    print(f"  Total images     : {total_images}")
    print(f"  Total time       : {total_time:.3f} s")
    print(f"  Avg latency/batch: {avg_latency_ms:.2f} ms")
    print(f"  FPS (images/sec) : {fps:.1f}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastSCNNDimming model test & benchmark"
    )
    parser.add_argument(
        "--height", type=int, default=128,
        help="Input height (default: 128) [PROJECT DECISION]",
    )
    parser.add_argument(
        "--width", type=int, default=224,
        help="Input width (default: 224) [PROJECT DECISION]",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Building FastSCNNDimming")
    model = FastSCNNDimming().to(device)

    total, trainable = count_parameters(model)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Model size (approx)  : {total * 4 / 1024 / 1024:.2f} MB (FP32)")

    # --- Shape test ---
    print(f"\n--- Shape Test (eval mode) ---")
    model.eval()
    x = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    with torch.inference_mode():
        out = model(x)
    print(f"  output: {list(out.shape)}")
    assert out.shape == (args.batch_size, 1, args.height, args.width), (
        f"Expected [{args.batch_size}, 1, {args.height}, {args.width}], "
        f"got {list(out.shape)}"
    )
    print("  ✓ shape test passed")

    # --- Debug feature shapes ---
    print(f"\n--- Debug Feature Shapes ---")
    with torch.inference_mode():
        feats = model.forward_debug(x)
    for name, tensor in feats.items():
        print(f"  {name}: {list(tensor.shape)}")

    # --- Benchmark ---
    _benchmark(
        model,
        device,
        args.height,
        args.width,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
