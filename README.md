# Fast-SCNN Dimming: Foreground Protection / Dimming Soft Mask Prediction

## 📐 Architectures

The project supports two main architectures:

### 1. Single-Head Model (`fast_scnn_dimming`)
An optimized Fast-SCNN structure with a single classifier head outputting a full-resolution soft mask.
* **Learning to Downsample (LtD)**:
  * Layer 1: Standard 3×3 Conv (stride 2) $\to$ $H/2$
  * Layer 2: 3×3 DSConv (stride 2) $\to$ $H/4$
  * Layer 3: 3×3 DSConv (stride 2) $\to$ $H/8$ (Original Fast-SCNN stride 2)

### 2. Dual-Head Model (`fast_scnn_dual_head`)
A **Coarse-to-Fine** output stage designed to segment complex objects with high-frequency boundaries (such as fine feathers or hair):
* **Coarse Head**: Performs global semantic localization at low resolution ($H/8$) on a 2x downsampled image.
* **Sigmoid Detach (Stop-Gradient)**: The low-resolution coarse mask is converted to a spatial prompt using Sigmoid and detached (`.detach()`) to prevent Fine Head gradients from destabilizing global positioning.
* **Fine Head (Refinement)**: A multi-scale decoder that receives the Coarse Prompt along with high-resolution skip connections ($H/2$ and $H/4$) directly from the LtD stage to reconstruct sharp details.
* **Single-pass Mode (`resolution_hierarchy=False`)**: The backbone is run only once on the full-resolution image, sharing feature representations between both heads to reduce computational latency (FLOPs).

---

## 📂 Dataset Format

Ensure your custom dataset is structured as follows:

```
dataset_root/
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

### Soft Target Generation
During data loading, each sample returns a **binary mask** (original foreground) and a **soft mask** (expanded protection zone with cosine feathering):
1. **Dilation**: Expands the foreground by `protection_radius`.
2. **Feathering**: Creates a smooth transition region of `transition_width` using a cosine decay function.

---

## 🖥️ GPU Selection on Multi-GPU Server (gportal2)

If you are running on a shared multi-GPU server like **gportal2**:

1. Check current GPU utilization to see which GPU is idle by running:
   ```bash
   nvidia-smi
   ```
2. Specify which GPU to use by prepending `CUDA_VISIBLE_DEVICES=id` (where `id` is the index of the free GPU, e.g., `0` or `1`) when executing the scripts, or modify the variable directly in the shell scripts (`train.sh`, `test.sh`, `eval.sh`):
   ```bash
   # Run training on GPU 0
   CUDA_VISIBLE_DEVICES=0 ./train.sh

   # Run evaluation on GPU 1
   CUDA_VISIBLE_DEVICES=1 ./eval.sh
   ```

---

## 🚀 Getting Started

Quickly run training, inference, and evaluation using the pre-configured shell scripts:

### 1. Training (`train.sh`)
Start training on your dataset using `train.sh`:
```bash
./train.sh
```

**Key Parameters for `train.py`**:
* `--model`: Model type (`fast_scnn_dimming` or `fast_scnn_dual_head`).
* `--data-root`: Path to the dataset directory.
* `--pretrained`: Path to pretrained weights (`.pt`) to initialize the backbone.
* `--train-height` / `--train-width`: Spatial resolution of training inputs (e.g., 512 × 1024).
* `--val-height` / `--val-width`: Spatial resolution of validation inputs.
* `--batch-size`: Batch size (e.g., 16).
* `--epochs`: Total number of training epochs.
* `--optimizer`: Optimizer type (`adamw` or `sgd`).
* `--learning-rate` (or `--lr`): Initial learning rate (e.g., `1e-4` for fine-tuning, `1e-3` for training from scratch).
* `--scheduler`: LR scheduler (`poly` or `cosine`).
* `--poly-power`: Decay power factor for the Poly scheduler.
* `--checkpoint-save-interval`: Interval (in epochs) to save checkpoints.
* `--seed`: Random seed for reproducibility.
* `--allow-threshold`: Automatically thresholds masks that are not strictly binary.
* `--protection-radius`: Dilation radius for soft target generation.
* `--transition-width`: Cosine feathering width for soft target generation.
* `--coarse-only-epochs`: Number of epochs in **Phase 1** to train only the Coarse Head (default: `5`). Helps stabilize the backbone before fine-tuning details.
* `--coarse-joint-training`: If enabled, continues training the Coarse Head jointly with the Fine Head during **Phase 2**. If disabled, Coarse Head is frozen in Phase 2.
* `--coarse-edge-mask-kernel`: Kernel size for edge masking in the Coarse Head loss (helps the Coarse Head ignore edge details and focus on the core region).
* `--coarse-target-dilation-kernel`: Dilation kernel size applied to targets for the Coarse Head loss.

---

### 2. Batch Inference / Testing (`test.sh`)
Run predictions on a folder of images using `test.sh`:
```bash
./test.sh
```

**Key Parameters for `inference.py`**:
* `--weights`: Path to the trained model checkpoint (`.pt`).
* `--input`: Path to a single image or a folder of images.
* `--output-dir`: Directory to save visual results (heatmaps, dimmed previews, side-by-side grids).
* `--height` / `--width`: Input resolution to resize images during inference.
* `--resize-to-original`: Automatically rescales the output mask back to the original image dimensions.

---

### 3. Metric Evaluation (`eval.sh`)
Evaluate model performance and compute detailed metrics using `eval.sh`:
```bash
./eval.sh
```

**Key Parameters for `evaluate.py`**:
* `--weights`: Path to the model checkpoint (`.pt`).
* `--data-root`: Path to the evaluation dataset.
* `--split`: Target split (`val` or `test`).
* `--output-dir`: Directory to save the final metrics summary as a JSON file.
* `--val-height` / `--val-width`: Input resolution to use during evaluation.
* `--batch-size`: Batch size.
* `--threshold`: Binarization threshold (e.g., `0.9`) for binary IoU and Dice evaluations.
* `--allow-threshold`: Allow thresholding of ground truth masks.

---

## 📊 Metrics Description

Evaluation outputs a series of metric indicators to evaluate dimming quality:
1. **Binary Segmentation Metrics (at target `--threshold`)**:
   * `fg_iou` / `bg_iou`: Intersection over Union of foreground and background.
   * `miou`: Mean IoU, computed as `(fg_iou + bg_iou) / 2.0`.
   * `dice`, `precision`, `recall`: Classic segmentation indicators.
2. **Soft Mask Quality**:
   * `soft_mae` / `soft_mse`: Mean Absolute Error / Mean Squared Error between the predicted probability map and the soft target.
3. **Foreground Protection**:
   * `fg_mean_protection`: Average predicted value on original foreground pixels (closer to 1.0 is better).
   * `fg_under_protection_error`: Total under-protection penalty score.
4. **Far-Background Leakage**:
   * `far_bg_leakage`: Average predicted value on far-background pixels where the soft target is 0. Closer to 0.0 means more power-saving capability.

---

