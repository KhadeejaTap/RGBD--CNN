# RGB-D Depth Completion

## Paper Interpretation

This repository is my interpretation and implementation of the ideas
presented in:

> Yansong Du, Yutong Deng, Yuting Zhou, Feiyu Jiao, Jian Song, and Xun Guan.
> “Towards High-Precision Depth Sensing via Monocular-Aided iToF and RGB
> Integration.” arXiv:2508.16579, 2025.
> [https://arxiv.org/abs/2508.16579](https://arxiv.org/abs/2508.16579)

It is not intended to be an official reproduction of the authors' code or
results. The pipeline, model choices, and experiments here reflect my
understanding of the paper's problem formulation and proposed direction.

## Issue / Goal

The ToF depth data is sparse, contains invalid pixels, and is captured from a
different camera viewpoint than the RGB images. Directly using the raw depth
therefore leaves holes and can introduce geometric misalignment or misleading
zero-valued features.

The goal is to produce dense, RGB-guided depth estimates in the RGB camera
plane while preserving the reliability of valid ToF measurements and measuring
the remaining error against ground-truth depth.

## Solution

The project uses the following pipeline:

1. **Reproject ToF depth into the RGB plane**  
   `reproj.py` applies calibrated ToF intrinsics, RGB intrinsics, and
   ToF-to-RGB extrinsics. It filters invalid/range-limited measurements and
   uses a z-buffer so the nearest projected point wins at each RGB pixel.

2. **Build a matching validity mask**  
   `proj_mask.py` reuses the same reprojection operation to create
   `frame_*_proj_valid_mask.npy`. This keeps the depth values and validity
   information pixel-aligned.

3. **Complete sparse depth with RGB features**  
   `uhrig2.py` uses an RGB-D encoder-decoder model. The RGB branch uses a
   pretrained ResNet-18, while the depth branch uses sparsity-invariant
   convolutions that normalize by the number of valid input pixels. The
   branches are fused at multiple resolutions and decoded into a dense depth
   map.

4. **Train with geometry-aware objectives**  
   The model combines regression, structural distillation, edge-aware
   smoothness, and surface-normal consistency losses. Depth is normalized
   during prediction and converted back to millimeters for evaluation.

5. **Inspect results**  
   - `toferr.py` computes ToF-versus-ground-truth error metrics and saves error
     maps.
   - `debug.py` creates visualizations of monocular depth estimates.
   - `overlay.py` blends depth visualizations for comparison.
   - `uhrig.py` and `uhrig2.py` contain earlier and revised model experiments.

## Effect

The pipeline turns sparse, misaligned ToF measurements into RGB-aligned
training inputs and gives the model explicit information about which depth
pixels are trustworthy. Sparsity-invariant processing prevents missing depth
from being treated as real zero depth, while RGB features provide structure in
regions where ToF measurements are absent.

The scripts also report MAE, MSE, RMSE, and absolute-relative error, with
separate error summaries for valid and hole regions. This makes it possible to
compare whether a model improves depth completion rather than only producing
visually smoother images.

## Main Inputs and Outputs

Expected frame data is stored under `data/`:

- `frame_*_tof_mm.npy`: raw ToF depth in millimeters
- `frame_*_valid_mask.npy`: raw ToF validity mask
- `frame_*_rgb.png`: RGB frame
- `frame_*_gt_mm.npy`: ground-truth depth
- `frame_*_mde.npy`: monocular depth estimate

Generated data includes:

- `frame_*_depth_proj_mm.npy`: ToF depth reprojected into the RGB plane
- `frame_*_proj_valid_mask.npy`: projected validity mask
- `frame_*_pred_cconv.png`: dense model prediction visualization
- `frame_*_error_cconv.png`: prediction error visualization

## Example Commands

```bash
# Reproject ToF depth into the RGB plane
python reproj.py --input-dir data --out-dir data

# Generate projected validity masks
python proj_mask.py --input-dir data --out-dir data

# Run the sparse RGB-D experiment
python uhrig2.py

# Compare raw ToF depth with ground truth
python toferr.py --dir data
```

The model scripts currently run as experiments and use the repository's
relative `data/` paths. PyTorch, torchvision, NumPy, Pillow, and Matplotlib
are required.

work done as a Research Assistant at SAIL
