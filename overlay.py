from PIL import Image
import os
from glob import glob
def overlay(in_dir, out_dir):
	os.makedirs(out_dir, exist_ok=True)
	mde = sorted(glob(os.path.join(in_dir, "frame_*_mde_vis.png")))
	proj = sorted(glob(os.path.join(in_dir, "frame_*_proj_vis.png")))
	proj_map = {os.path.basename(p).replace("_proj_vis.png", ""): p for p in proj}

	for m in mde:
		key = os.path.basename(m).replace("_mde_vis.png", "")
		p = proj_map.get(key)
		if p is None:
			print("no match")
			continue
		img1 = Image.open(m).convert("RGBA")
		img2 = Image.open(p).convert("RGBA")
		res = Image.blend(img1, img2, alpha=0.5)
		out_path = os.path.join(out_dir, f"{key}_overlay.png")
		res.save(out_path)
	print("done breh")

if __name__ == "__main__":
	overlay("mde_vis", "mde_vis")
