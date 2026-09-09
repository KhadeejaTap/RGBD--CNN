import numpy as np

TOF_BASE_WIDTH, TOF_BASE_HEIGHT = 640, 480
RGB_BASE_WIDTH, RGB_BASE_HEIGHT = 1920, 1080

# Precise fx from calibration
# ToF fx = 544.462653
# RGB fx = 910.799450
K_TOF_BASE = np.array([
    [544.462653, 0, (TOF_BASE_WIDTH - 1) / 2.0],
    [0, 544.462653, (TOF_BASE_HEIGHT - 1) / 2.0],
    [0, 0, 1]
], dtype=np.float64)

K_RGB_BASE = np.array([
    [910.799450, 0, (RGB_BASE_WIDTH - 1) / 2.0],
    [0, 910.799450, (RGB_BASE_HEIGHT - 1) / 2.0],
    [0, 0, 1]
], dtype=np.float64)

# Real-world Extrinsics from camera_and_R_T_matrices.yml
# R is the rotation from ToF to RGB
R = np.array([
    [ 0.9998933213423421, -0.001586456582046135,  0.014519954906713411],
    [ 0.0014369365883950588, 0.99994589849845628,  0.010302198277837599],
    [-0.014535513345618034, -0.010280234998687087, 0.99984150524978266]
], dtype=np.float64)

# T is the translation from ToF to RGB (in meters)
T = np.array([0.0020168806248490076, 0.070622275765060152, -0.013673635196481401], dtype=np.float64)

def _scale_intrinsics(base_k: np.ndarray, base_width: int, base_height: int, width: int, height: int) -> np.ndarray:
    scale_x = width / base_width
    scale_y = height / base_height
    k = base_k.copy()
    k[0, 0] *= scale_x
    k[0, 2] *= scale_x
    k[1, 1] *= scale_y
    k[1, 2] *= scale_y
    return k


def project_depth_ideal(depth_mm, valid_mask, quantize_mm=0, rgb_width=RGB_BASE_WIDTH, rgb_height=RGB_BASE_HEIGHT):
    """
    Reprojects ToF depth into RGB plane using ideal intrinsics and hardcoded extrinsics.

    Args:
        depth_mm: (H, W) array of ToF Euclidean depth in millimeters.
        valid_mask: (H, W) boolean array marking valid ToF pixels.
        quantize_mm: Quantization step in mm (default 0 = no quantization, matching real-world behavior).

    Returns:
        (RGB_H, RGB_W) array of reprojected depth in millimeters.
    """
    # 1. Prepare coordinates
    Z = depth_mm.astype(np.float64) / 1000.0  # Convert to meters
    h, w = Z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    K_tof = _scale_intrinsics(K_TOF_BASE, TOF_BASE_WIDTH, TOF_BASE_HEIGHT, w, h)
    K_rgb = _scale_intrinsics(K_RGB_BASE, RGB_BASE_WIDTH, RGB_BASE_HEIGHT, rgb_width, rgb_height)

    # Valid mask merging: (Range + Finite) AND User-provided mask
    internal_mask = (depth_mm >= 300.0) & (depth_mm <= 8333.0) & np.isfinite(Z)
    combined_mask = internal_mask & valid_mask.astype(bool)

    if not np.any(combined_mask):
        return np.zeros((rgb_height, rgb_width), dtype=np.float32)

    # 2. Unproject to 3D ToF Space
    K_inv = np.linalg.inv(K_tof)
    pixels = np.stack((u[combined_mask], v[combined_mask], np.ones_like(u[combined_mask])), axis=0)
    points_tof = (K_inv @ pixels) * Z[combined_mask]

    # 3. Transform to RGB Space
    # P_rgb = R * P_tof + T
    points_rgb = (R @ points_tof) + T[:, np.newaxis]

    # 4. Project to RGB Pixels
    # u_rgb = K_rgb * P_rgb / Z_rgb
    pixels_rgb_h = K_rgb @ points_rgb
    u_rgb = pixels_rgb_h[0] / pixels_rgb_h[2]
    v_rgb = pixels_rgb_h[1] / pixels_rgb_h[2]
    z_rgb = points_rgb[2]

    # 5. Filter valid pixels in RGB plane
    in_bounds = (
        (u_rgb >= 0) & (u_rgb < rgb_width) &
        (v_rgb >= 0) & (v_rgb < rgb_height) &
        (z_rgb > 0)
    )

    u_idx = np.clip(np.round(u_rgb[in_bounds]).astype(np.int32), 0, rgb_width - 1)
    v_idx = np.clip(np.round(v_rgb[in_bounds]).astype(np.int32), 0, rgb_height - 1)
    z_vals = z_rgb[in_bounds].astype(np.float32)

    # 6. Create Z-Buffer (handle occlusion by taking minimum depth per pixel)
    depth_proj = np.full((rgb_height, rgb_width), np.inf, dtype=np.float32)
    flat_idx = v_idx * rgb_width + u_idx
    np.minimum.at(depth_proj.ravel(), flat_idx, z_vals)

    # 7. Final processing (mm conversion, quantization)
    depth_mm_out = np.zeros((rgb_height, rgb_width), dtype=np.float32)
    valid_proj = np.isfinite(depth_proj)
    depth_mm_out[valid_proj] = depth_proj[valid_proj] * 1000.0

    if quantize_mm > 0:
        mask = depth_mm_out > 0
        depth_mm_out[mask] = np.round(depth_mm_out[mask] / quantize_mm) * quantize_mm

    # Convert to uint16 (matching real-world process_captured_frames.py)
    # uint16 range: 0-65535 mm (0-65.535 m)
    depth_uint16 = np.clip(depth_mm_out, 0, 65535).astype(np.uint16)
    return depth_uint16

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Ideal Reprojection (No Calibration File Required)")
    parser.add_argument("--input-dir", type=str, required=True, help="Dir with frame_XXXX_tof_mm.npy")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--quantize-mm", type=float, default=0.25, help="Quantization (mm)")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(input_dir.glob("frame_*_tof_mm.npy"))
    print(f"Ideal Reprojection: Processing {len(depth_files)} frames...")

    for fpath in depth_files:
        frame_num = fpath.stem.split("_")[1]
        depth_mm = np.load(fpath)

        mask_path = input_dir / f"frame_{frame_num}_valid_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing validity mask: {mask_path}")

        valid_mask = np.load(mask_path)

        proj_mm = project_depth_ideal(depth_mm, valid_mask, args.quantize_mm)

        out_path = out_dir / f"frame_{frame_num}_depth_proj_mm.npy"
        np.save(out_path, proj_mm.astype(np.uint16))

    print(f"Done! Results in {args.out_dir}")
