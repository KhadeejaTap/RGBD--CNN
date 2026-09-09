import argparse
from pathlib import Path
import numpy as np

# import your existing reprojection function - update this import to match
# whatever filename your reproj script is actually saved as
from reproj import project_depth_ideal

parser = argparse.ArgumentParser(description="Build reprojected valid masks from ToF depth + valid mask")
parser.add_argument("--input-dir", type=str, default="data", help="Dir with frame_XXXX_tof_mm.npy and frame_XXXX_valid_mask.npy")
parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
parser.add_argument("--rgb-width", type=int, default=960, help="Half-res RGB width you're reprojecting into")
parser.add_argument("--rgb-height", type=int, default=540, help="Half-res RGB height you're reprojecting into")
parser.add_argument("--quantize-mm", type=float, default=0.25, help="Must match value used for depth reprojection")

args = parser.parse_args()
input_dir = Path(args.input_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

depth_files = sorted(input_dir.glob("frame_*_tof_mm.npy"))
print(f"Building projected valid masks for {len(depth_files)} frames at {args.rgb_width}x{args.rgb_height}...")

for fpath in depth_files:
	frame_num = fpath.stem.split("_")[1]
	depth_mm = np.load(fpath)

	mask_path = input_dir / f"frame_{frame_num}_valid_mask.npy"
	if not mask_path.exists():
		raise FileNotFoundError(f"Missing validity mask: {mask_path}")
	valid_mask = np.load(mask_path)

	# reuse the exact same z-buffered projection used for depth itself,
	# so mask and depth agree pixel-for-pixel on what counts as "valid"
	proj_mm = project_depth_ideal(
		depth_mm, valid_mask,
		quantize_mm=args.quantize_mm,
		rgb_width=args.rgb_width,
		rgb_height=args.rgb_height,
	)

	proj_valid_mask = (proj_mm > 0)

	out_path = out_dir / f"frame_{frame_num}_proj_valid_mask.npy"
	np.save(out_path, proj_valid_mask)

print(f"Done! Results in {args.out_dir}")
