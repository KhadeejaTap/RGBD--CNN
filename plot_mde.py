import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def save_mde_visualizations(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    npy_files = sorted(glob(os.path.join(input_dir, "frame_*_mde.npy")))
    for npy_path in npy_files:
        depth = np.load(npy_path)
        basename = os.path.basename(npy_path).replace("_mde.npy", "_mde_vis.png")
        out_path = os.path.join(output_dir, basename)

        fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(depth, cmap="turbo")
        ax.axis("off")
        fig.savefig(out_path, dpi=100, transparent=True)
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    input_dir = "data"
    output_dir = "mde_vis"
    save_mde_visualizations(input_dir, output_dir)
