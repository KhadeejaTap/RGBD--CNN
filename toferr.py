import numpy as np
import argparse
from glob import glob
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='dir with the tof and gt npys')
args = parser.parse_args()

tof_files = sorted(glob(f"{args.dir}/*tof_mm.npy"))
gt_files = sorted(glob(f"{args.dir}/*gt_mm.npy"))

assert len(tof_files) == len(gt_files), "mismatch in num files of tof vs gt"
assert len(tof_files) > 0, "no files found"
out_dir = "error_maps"
for tof_path, gt_path in zip(tof_files, gt_files):
	tof = np.load(tof_path)
	gt = np.load(gt_path)

	valid = ~np.isnan(tof)

	tof_valid = tof[valid]
	gt_valid = gt[valid]

	mae = np.mean(np.abs(tof_valid - gt_valid))
	mse = np.mean((tof_valid - gt_valid)**2)
	rmse = np.sqrt(mse)
	abs_rel = np.mean(np.abs(tof_valid - gt_valid) / gt_valid)

	frame_id = os.path.basename(tof_path).split('_tof')[0]
	print(f"{frame_id}: MAE={mae:.4f} MSE={mse:.4f} RMSE={rmse:.4f} AbsRel={abs_rel:.4f} "
		f"(valid pixels: {valid.sum()}/{valid.size})")

	# full-frame error map, keeping shape, marking invalid (hole) pixels as NaN
	# so they show up blank/transparent instead of a fake zero-error patch
	error_map = np.abs(tof - gt)
	error_map[~valid] = np.nan

	fig, ax = plt.subplots(figsize=(10, 8))
	im = ax.imshow(error_map, cmap='hot')
	ax.set_title(f"{frame_id} Error Map (tof vs gt)")
	plt.colorbar(im, ax=ax)
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, f"{frame_id}_error.png"), dpi=200)
	plt.close(fig)

print(f"\nsaved {len(tof_files)} error maps to {out_dir}/")
