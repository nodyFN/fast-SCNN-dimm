# Fast-SCNN Dimming: Foreground Protection / Dimming Soft Mask Prediction

## Task Definition

> **This is NOT Alpha Matting.**

This project predicts a **foreground protection mask** for TV SoC background dimming:

```
M(x, y) ∈ [0, 1]

M = 1: foreground — fully protected brightness
M = 0: far background — maximum dimming allowed
0 < M < 1: transition / safety region
```

The output represents **foreground protection strength / dimming protection strength**, not physical alpha transparency.

### Use Case

1. Identify foreground / salient objects that need brightness protection
2. Foreground core → mask ≈ 1 (preserve brightness)
3. Smooth transition zone around foreground
4. Far background → mask ≈ 0 (allow dimming)
5. Convert protection mask to brightness/dimming map for TV power reduction

---

## Architecture

Modified Fast-SCNN with higher-resolution feature maps for better mask detail.

```
Input [B, 3, 128, 224]                              ← [PROJECT DECISION] landscape TV
        │
        ▼
Learning to Downsample
  Layer 1: 3×3 Conv s=2      → [B, 32, 64, 112]     /2
  Layer 2: 3×3 DSConv s=2    → [B, 48, 32, 56]      /4
  Layer 3: 3×3 DSConv s=1    → [B, 64, 32, 56]      /4  ← [PROJECT DECISION] was s=2 in original
        │
        ├──── shallow skip [B, 64, 32, 56] ──────────┐
        │                                             │
        ▼                                             │
Global Feature Extractor                              │
  Stage 1: 64→64   t=6 n=3 s=2  → [B, 64, 16, 28]   /8
  Stage 2: 64→96   t=6 n=3 s=2  → [B, 96, 8, 14]    /16 ← [PROJECT DECISION] was /32
  Stage 3: 96→128  t=6 n=3 s=1  → [B, 128, 8, 14]   /16
        │                                             │
        ▼                                             │
Pyramid Pooling Module (1, 2, 3, 6)                   │
  [B, 128, 8, 14] → [B, 128, 8, 14]                  │
        │                                             │
        ▼                                             │
Feature Fusion Module ◄───────────────────────────────┘
  deep:    bilinear upsample 8×14 → 32×56
           → DW Conv → BN → ReLU → 1×1 PW → BN
  shallow: 1×1 Conv → BN
  fused:   ReLU(high + low) → [B, 128, 32, 56]
        │
        ▼
Classifier
  DSConv 128→128
  DSConv 128→128
  Dropout(0.1)                                        ← [PROJECT DECISION]
  1×1 Conv 128→1
  → [B, 1, 32, 56]
        │
        ▼
Bilinear Upsample → [B, 1, 128, 224]
        │
        ▼
Raw logits (no sigmoid in model)
        │
        ▼
torch.sigmoid() → Protection Mask [0, 1]
```

### Key Architecture Changes from Original Fast-SCNN

| Change | Original | This Project | Rationale |
|---|---|---|---|
| LtD output stride | /8 | /4 | Higher-res skip for mask detail |
| Deep output stride | /32 | /16 | Better spatial preservation |
| Output channels | 2 (classes) | 1 (BCE logit) | Binary task |
| Input resolution | varies | 128×224 | Landscape TV |

---

## Input Resolution Convention

```python
# [PROJECT DECISION] Landscape TV image
width  = 224    # horizontal
height = 128    # vertical

# PyTorch tensor: [B, C, H, W]
tensor_shape = [B, 3, 128, 224]
```

**Do not swap H and W.** All configs use explicit `train_height`/`train_width`.

---

## Dataset Format

```
duts_data/
├── train/
│   ├── images/    (*.jpg, *.png)
│   └── masks/     (*.png — single-channel, {0,1} or {0,255})
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Images and masks are paired by file stem. Prepare DUTS:

```bash
python prepare_duts_dataset.py --src /path/to/DUTS --dest duts_data
```

### Mask Value Handling

| Mask values | Behavior |
|---|---|
| {0, 1} | Used directly |
| {0, 255} | 255 → 1 |
| Other | Error (use `--allow-threshold` to threshold at 127) |

Check your masks:
```bash
python check_mask_value.py --data-root duts_data
```

---

## Soft Target Generation

Each dataset sample returns **two** targets:

1. **`binary_mask`**: Original {0, 1} foreground GT
2. **`soft_mask`**: Foreground-preserving dimming soft target

### Algorithm

```
Step A: Protection Dilation
  dilate(binary_mask, ellipse kernel, radius=2)
  → protected_fg region, all M=1

Step B: Outward Distance Transform
  cv2.distanceTransform on inverted protected_fg
  → distance of each background pixel to nearest protected foreground

Step C: Cosine Feather (transition_width=8)
  protected foreground:  M = 1
  transition (0 < d < T): M = 0.5 × (1 + cos(π × d / T))
  far background (d ≥ T): M = 0
```

### Properties

- Original foreground → `soft_target == 1.0` (guaranteed)
- Dilated protection zone → `soft_target == 1.0`
- Smooth cosine transition outward
- Far background → `soft_target == 0.0`

### Augmentation Ordering

```
load RGB + binary mask
    ↓
joint geometric augmentation (scale, crop, flip)
    ↓
resize to 128×224 (mask: nearest-neighbor)
    ↓
generate soft target at FINAL resolution
    ↓
ImageNet normalize
```

### Configurable Parameters

```python
protection_radius = 2     # [PROJECT DECISION] dilation radius
transition_width = 8      # [PROJECT DECISION] cosine feather width
soft_target_mode = "cosine"
```

### Visual Inspection

```bash
python inspect_soft_target.py --data-root duts_data --num-samples 10

# Compare parameters
python inspect_soft_target.py --data-root duts_data --num-samples 5 --compare
```

---

## Loss

```
L_total = λ_bce × L_bce + λ_l1 × L_l1 + λ_protect × L_protect
```

| Component | Formula | Default λ |
|---|---|---|
| **BCE** | `BCEWithLogitsLoss(logits, soft_target)` | 1.0 |
| **L1** | `L1(sigmoid(logits), soft_target)` | 1.0 |
| **Foreground Protection** | `sum((1-prob) × binary_mask) / (sum(binary_mask) + ε)` | 2.0 |

[PROJECT DECISION] `λ_protect = 2.0` — foreground under-protection is penalized more heavily than background over-protection.

---

## Metrics

### Binary Segmentation (threshold=0.5)
- Foreground IoU, Dice, Precision, Recall

### Soft Mask Quality
- MAE, MSE (pred prob vs soft target)

### Foreground Protection
- **Mean foreground protection**: `mean(prob[binary_gt == 1])` — closer to 1 is better
- **Under-protection error**: `mean(1 - prob[binary_gt == 1])` — lower is better
- **Under-protection rate @0.9**: ratio of `prob < 0.9` in foreground — lower is better

### Far-Background Leakage
- `mean(prob[soft_target == 0])` — only far background, excluding transition

---

## Training

```bash
# Default training
python train.py --data-root duts_data

# Full customization
python train.py \
  --data-root duts_data \
  --train-height 128 \
  --train-width 224 \
  --batch-size 16 \
  --epochs 200 \
  --learning-rate 1e-3 \
  --lambda-bce 1.0 \
  --lambda-l1 1.0 \
  --lambda-protect 2.0 \
  --protection-radius 2 \
  --transition-width 8 \
  --seed 42

# Smoke test (no dataset needed)
python train.py --smoke-test
```

### Checkpoints

Saved to `checkpoints/<timestamp>/`:
- `latest.pt` — always saved
- `best_val_loss.pt` — best validation loss
- `best_fg_protection.pt` — best foreground mean protection
- `best_soft_mae.pt` — best soft MAE

Periodic saving: `--checkpoint-save-interval N` (save every N epochs)

### Resume Training

```bash
python train.py --data-root duts_data --resume checkpoints/<timestamp>/latest.pt
```

### TensorBoard

```bash
tensorboard --logdir runs/
```

---

## Evaluation

```bash
python evaluate.py \
  --weights checkpoints/<timestamp>/best_val_loss.pt \
  --data-root duts_data

# Evaluate on test split
python evaluate.py \
  --weights checkpoints/<timestamp>/best_val_loss.pt \
  --data-root duts_data \
  --split test
```

Outputs JSON summary to `evaluation_results/`.

---

## Inference

```bash
# Single image
python inference.py \
  --weights checkpoints/<timestamp>/best_val_loss.pt \
  --input photo.jpg

# Folder
python inference.py \
  --weights checkpoints/<timestamp>/best_val_loss.pt \
  --input images/ \
  --resize-to-original
```

Outputs: soft mask, binary visualization, heatmap, dimmed preview, side-by-side comparison.

---

## ONNX Export

```bash
# Export raw logits (default)
python export.py --weights checkpoints/<timestamp>/best_val_loss.pt

# Export with sigmoid
python export.py --weights checkpoints/<timestamp>/best_val_loss.pt --include-sigmoid
```

Default opset: 17 [PROJECT DECISION]. Validates shape and numerical consistency with PyTorch.

---

## Tests

```bash
pytest tests/ -v
```

| Test file | What it tests |
|---|---|
| `test_model.py` | Feature shapes at every stage, H/W checks, gradient flow |
| `test_soft_target.py` | Range, foreground preservation, direction, edge cases |
| `test_losses.py` | Loss behavior, empty foreground, backward pass |
| `test_dataset.py` | Mask formats, output shapes, binary/soft sync |
| `test_metrics.py` | Metric correctness with known inputs |

---

## Dimming Simulation

```python
# Brightness mapping: D = D_min + (1 - D_min) × M
# Example with D_min = 0.5:
#   M=1.0 → brightness 100%
#   M=0.8 → 90%
#   M=0.5 → 75%
#   M=0.0 → 50%
```

[PROJECT DECISION] This is a visualization aid, not a real TV power model.

---

## Project Structure

```
fast-scnn-dimming/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py                    # Centralized config (dataclass + CLI)
├── dataset.py                   # DimmingDataset + DataLoader factory
├── train.py                     # Training pipeline
├── evaluate.py                  # Evaluation with full metrics
├── inference.py                 # Single image / folder inference
├── export.py                    # ONNX export + validation
├── prepare_duts_dataset.py      # DUTS split utility
├── check_mask_value.py          # Mask value distribution checker
├── inspect_soft_target.py       # Soft target visual inspection
├── models/
│   ├── __init__.py
│   └── fast_scnn_dimming.py     # FastSCNNDimming model
├── utils/
│   ├── __init__.py
│   ├── losses.py                # BCE + L1 + FG Protection
│   ├── metrics.py               # All evaluation metrics
│   ├── soft_target.py           # Soft target generator
│   ├── scheduler.py             # PolyLR + CosineAnnealing
│   ├── checkpoint.py            # Atomic save/load
│   ├── visualization.py         # Dimming visualization
│   └── seed.py                  # Reproducibility
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_soft_target.py
│   ├── test_losses.py
│   ├── test_dataset.py
│   └── test_metrics.py
├── duts_data/                   # Dataset (gitignored content)
├── checkpoints/
├── training_results/
├── evaluation_results/
├── inference_results/
├── runs/                        # TensorBoard
└── exports/                     # ONNX models
```

---

## Ablation Plan

| Exp | Architecture | Target | Loss | Purpose |
|---|---|---|---|---|
| 0 | Original Fast-SCNN (OS8/OS32) | binary GT | BCE | Baseline reference |
| 1 | deep OS32→OS16 | binary GT | BCE | Does deeper resolution help? |
| 2 | shallow OS8→OS4, deep OS16 | binary GT | BCE | Does higher-res skip help? |
| 3 | OS4/OS16 | soft foreground-preserving GT | BCE | Does soft target help? |
| 4 | OS4/OS16 | soft GT | BCE + L1 | Does L1 improve quality? |
| **5** | **OS4/OS16** | **soft GT** | **BCE + L1 + FG Protect** | **Full baseline** |

Current implementation = **Exp 5**. Disable components via config:

```bash
# Reproduce Exp 3 (BCE only)
python train.py --lambda-l1 0 --lambda-protect 0

# Reproduce Exp 4 (BCE + L1)
python train.py --lambda-protect 0
```

---

## Future Work

The following are **NOT** implemented in V1 — reserved for future ablation:

- PPM `(1, 2, 4)` pool sizes
- Boundary / edge loss
- Attention (CBAM, SE)
- Temporal consistency for video
- EMA temporal smoothing
- Optical-flow-aware smoothing
- Hardware quantization (INT8)
- ASIC constraints for TV SoC
- Guided filter post-processing
- Actual TV power model
- Knowledge distillation
- Dice / Focal / Tversky loss
- ASPP module
- Dual-head architecture

---

## [PROJECT DECISION] Summary

All decisions not from the original Fast-SCNN paper are marked with `[PROJECT DECISION]`:

- Input: W=224 × H=128 (landscape TV)
- LtD output stride: /4 (not /8)
- Deep output stride: /16 (not /32)
- PPM pool sizes: (1, 2, 3, 6)
- Single-channel output with BCE
- Soft target: cosine feather with dilation
- Protection radius: 2, transition width: 8
- Loss: BCE(1.0) + L1(1.0) + FG Protect(2.0)
- Optimizer: AdamW (paper uses SGD)
- Dropout: 0.1
