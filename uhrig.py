import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from glob import glob
from PIL import Image
import math
import matplotlib.pyplot as plt
import os
torch.manual_seed(42)
np.random.seed(42)
# 1 prep data ie data loader
class RGBDDataset(Dataset):
	def __init__(self):
		#folder is data/ in cur dir
		self.data_dir = 'data'
		# rgb is *rgb.png
		self.rgb_files = sorted(glob(f"{self.data_dir}/*rgb.png"))
		#depth is *tof_mm.npy
		self.depth_files = sorted(glob(f"{self.data_dir}/*depth_proj_mm.npy"))
		# gt is *gt_mm.npy
		self.gt_files = sorted(glob(f"{self.data_dir}/*gt_mm.npy"))
		#mde
		self.mde_files = sorted(glob(f"{self.data_dir}/*mde.npy"))
		self.mask_files = sorted(glob(f"{self.data_dir}/*proj_valid_mask.npy"))
		assert len(self.rgb_files) == len(self.depth_files) == len(self.gt_files) == len(self.mde_files)
		self.n_samples = len(self.depth_files)
	def __getitem__(self, index):
		img = Image.open(self.rgb_files[index])
		img_transform = torchvision.transforms.ToTensor()
		rgb = img_transform(img)
		normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		rgb = normalize(rgb)

		depth = torch.from_numpy(np.load(self.depth_files[index])).unsqueeze(0)
		depth = torch.nan_to_num(depth, nan=0.0)  # still zero NaN so tensor is usable everywhere else;
		# SparseConv2d below also multiplies by mask directly, so invalid pixels are excluded
		# from the conv's weighted average regardless of what placeholder value sits here.
		gt = torch.from_numpy(np.load(self.gt_files[index])).unsqueeze(0)
		mde = torch.from_numpy(np.load(self.mde_files[index])).unsqueeze(0)
		mask = torch.from_numpy(np.load(self.mask_files[index])).unsqueeze(0).float()
		return rgb, depth, gt, mde, mask
	def __len__(self):
		return self.n_samples
class RGBEncoder(nn.Module):
	def __init__(self, pretrained=True):
		super(RGBEncoder, self).__init__()
		weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
		resnet = torchvision.models.resnet18(weights = weights)
		self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
		self.l1 = resnet.layer1
		self.l2 = resnet.layer2
		self.l3 = resnet.layer3
		self.l4 = resnet.layer4
	def forward(self, rgb):
		#layers go here?
		out = self.stem(rgb)
		out1 = self.l1(out)
		out2 = self.l2(out1)
		out3 = self.l3(out2)
		out4 = self.l4(out3)
		return out1, out2, out3, out4

class SparseConv2d(nn.Module):
	# Uhrig et al. sparsity-invariant conv: normalizes each output by how many
	# valid (masked-in) input pixels contributed to it, so invalid/zeroed pixels
	# don't get silently blended in as if they were real depth=0 signal.
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
			stride=stride, padding=padding, bias=False)
		# fixed (non-trainable) all-ones conv just to COUNT valid pixels in each window
		self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding, bias=False)
		nn.init.constant_(self.mask_conv.weight, 1.0)
		for p in self.mask_conv.parameters():
			p.requires_grad = False
		self.bias = nn.Parameter(torch.zeros(out_channels))

	def forward(self, x, mask):
		x = x * mask  # zero out invalid pixels explicitly before conv
		out = self.conv(x)
		with torch.no_grad():
			valid_count = self.mask_conv(mask)  # how many valid pixels fed each output position
		valid_count_safe = torch.clamp(valid_count, min=1e-5)
		out = out / valid_count_safe  # renormalize: average over VALID pixels only
		out = out + self.bias.view(1, -1, 1, 1)
		new_mask = (valid_count > 0).float()  # propagate validity to next layer
		return out, new_mask


class DepthEncoder(nn.Module):
	def __init__(self):
		super(DepthEncoder, self).__init__()
		resnet = torchvision.models.resnet18()
		self.sparse_conv1 = SparseConv2d(1, 64, kernel_size=7, stride=2, padding=3)
		self.bn1 = resnet.bn1
		self.relu = resnet.relu
		self.maxpool = resnet.maxpool
		self.l1 = resnet.layer1
		self.l2 = resnet.layer2
		self.l3 = resnet.layer3
		self.l4 = resnet.layer4

	def forward(self, depth, mask):
		# sparse-aware first conv: only layer that explicitly knows which pixels were real
		out, mask = self.sparse_conv1(depth, mask)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.maxpool(out)
		# NOTE: l1-l4 are standard resnet blocks, no further mask propagation past this point.
		# This fixes the worst offender (raw invalid pixels leaking into first conv) but
		# doesn't make the whole encoder sparsity-aware end to end. Revisit if roughness persists.
		out1 = self.l1(out)
		out2 = self.l2(out1)
		out3 = self.l3(out2)
		out4 = self.l4(out3)
		return out1, out2, out3, out4

class Fusion(nn.Module):
	def forward(self, rgb_feats, depth_feats):
		fused1 = torch.cat([rgb_feats[0], depth_feats[0]], dim=1)
		fused2 = torch.cat([rgb_feats[1], depth_feats[1]], dim=1)
		fused3 = torch.cat([rgb_feats[2], depth_feats[2]], dim=1)
		fused4 = torch.cat([rgb_feats[3], depth_feats[3]], dim=1)
		return fused1, fused2, fused3, fused4

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		# each stage: 3x3 conv to shrink channels, then upsample, then it gets concatenated with next skip connection
		self.dec4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
		self.dec3 = nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1)
		self.dec2 = nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1)
		self.dec1 = nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1)
		self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
		self.tanh = nn.Tanh()

	def forward(self, fused1, fused2, fused3, fused4):
		x = self.dec4(fused4)
		x = F.interpolate(x, size=fused3.shape[2:], mode='bilinear', align_corners=False)
		x = torch.cat([x, fused3], dim=1)

		x = self.dec3(x)
		x = F.interpolate(x, size=fused2.shape[2:], mode='bilinear', align_corners=False)
		x = torch.cat([x, fused2], dim=1)

		x = self.dec2(x)
		x = F.interpolate(x, size=fused1.shape[2:], mode='bilinear', align_corners=False)
		x = torch.cat([x, fused1], dim=1)

		x = self.dec1(x)
		x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

		x = self.final(x)
		x = self.tanh(x)
		x = F.interpolate(x, size=gt.shape[2:], mode='bilinear', align_corners=False)
		return x

# losses
def regression_loss(pred, gt):
	return F.smooth_l1_loss(pred, gt)

def structure_distillation_loss(pred, mde, window_size=11):
	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
	mu_mde = F.avg_pool2d(mde, window_size, stride=1, padding=window_size//2)

	mu_pred_sq = mu_pred ** 2
	mu_mde_sq = mu_mde ** 2
	mu_pred_mde = mu_pred * mu_mde

	sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
	sigma_mde_sq = F.avg_pool2d(mde ** 2, window_size, stride=1, padding=window_size//2) - mu_mde_sq
	sigma_pred_mde = F.avg_pool2d(pred * mde, window_size, stride=1, padding=window_size//2) - mu_pred_mde

	ssim_map = ((2 * mu_pred_mde + C1) * (2 * sigma_pred_mde + C2)) / \
		((mu_pred_sq + mu_mde_sq + C1) * (sigma_pred_sq + sigma_mde_sq + C2))

	return (1 - ssim_map.mean()) / 2

def smoothness_loss(pred, rgb, alpha=10):
	pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
	pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

	rgb_dx = rgb[:, :, :, 1:] - rgb[:, :, :, :-1]
	rgb_dy = rgb[:, :, 1:, :] - rgb[:, :, :-1, :]

	weight_x = torch.exp(-alpha * rgb_dx.abs().mean(dim=1, keepdim=True))
	weight_y = torch.exp(-alpha * rgb_dy.abs().mean(dim=1, keepdim=True))

	smooth_x = (pred_dx.abs() * weight_x).mean()
	smooth_y = (pred_dy.abs() * weight_y).mean()

	return smooth_x + smooth_y

def normal_consistency_loss(pred, gt, grad_scale=10.0):
	def compute_normals(depth):
		dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
		dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
		dx = F.pad(dx, (0, 1, 0, 0))
		dy = F.pad(dy, (0, 0, 0, 1))
		normal = torch.cat([-dx * grad_scale, -dy * grad_scale, torch.ones_like(depth)], dim=1)
		normal = F.normalize(normal, dim=1)
		return normal

	pred_normal = compute_normals(pred)
	gt_normal = compute_normals(gt)

	cos_sim = (pred_normal * gt_normal).sum(dim=1)
	return (1 - cos_sim).mean()

# ---- data loading ----
dataset = RGBDDataset()
batch_size = 4
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers=3)
dataiter = iter(dataloader)
rgb, depth, gt, mde, mask = data = next(dataiter)

# ---- check raw data for NaN / bad values before anything touches them ----
print("rgb:", rgb.min().item(), rgb.max().item(), torch.isnan(rgb).any().item())
print("depth:", depth.min().item(), depth.max().item(), torch.isnan(depth).any().item())
print("gt:", gt.min().item(), gt.max().item(), torch.isnan(gt).any().item())
print("mde:", mde.min().item(), mde.max().item(), torch.isnan(mde).any().item())

# ---- encoders ----
rgb_encoder = RGBEncoder()
rgb_feats = rgb_encoder(rgb)
depth_encoder = DepthEncoder()
depth_feats = depth_encoder(depth, mask)

# ---- fusion ----
fusion = Fusion()
fused = fusion(rgb_feats, depth_feats)

# ---- decoder ----
decoder = Decoder()
out = decoder(fused[0], fused[1], fused[2], fused[3])
print("out:", out.min().item(), out.max().item(), torch.isnan(out).any().item())

# ---- loss ----
depth_min = 300.0
depth_max = 8333.0
gt_norm = 2 * (gt - depth_min) / (depth_max - depth_min) - 1
gt_norm = torch.clamp(gt_norm, -1.0, 1.0)  # safety clip
mde_norm = (mde - mde.amin(dim=(2,3), keepdim=True)) / \
	(mde.amax(dim=(2,3), keepdim=True) - mde.amin(dim=(2,3), keepdim=True) + 1e-8)
mde_norm = mde_norm * 2 - 1  # match pred's [-1,1] range
l_reg = regression_loss(out, gt_norm)
l_struct = structure_distillation_loss(out, mde_norm)
l_smooth = smoothness_loss(out, rgb)
l_normal = normal_consistency_loss(out, gt_norm)
total_loss = 1.0 * l_reg + 0.1 * l_smooth + 0.01 * l_struct + 0.1 * l_normal
print("individual losses: ", l_reg, l_struct, l_smooth, l_normal)
print("total:", total_loss.item())
optimizer = torch.optim.Adam(
	list(rgb_encoder.parameters()) +
	list(depth_encoder.parameters()) +
	list(decoder.parameters()),
	lr=1e-4
)

optimizer.zero_grad()
total_loss.backward()
optimizer.step()

print("backward done, loss:", total_loss.item())

# ---- overfit on single sample ----

# grab one sample directly, add batch dim back

rgb_s, depth_s, gt_s, mde_s, mask_s = dataset[0]
rgb_s = rgb_s.unsqueeze(0)
depth_s = depth_s.unsqueeze(0)
gt_s = gt_s.unsqueeze(0)
mde_s = mde_s.unsqueeze(0)
mask_s = mask_s.unsqueeze(0)

# depth already has NaN zeroed out in __getitem__; mask_s tells SparseConv2d which
# of those pixels were real vs just a NaN placeholder

depth_min = 300.0
depth_max = 8333.0
gt_s_norm = 2 * (gt_s - depth_min) / (depth_max - depth_min) - 1
gt_s_norm = torch.clamp(gt_s_norm, -1.0, 1.0)

n_overfit_iters = 100
for i in range(n_overfit_iters):
	rgb_feats = rgb_encoder(rgb_s)
	depth_feats = depth_encoder(depth_s, mask_s)
	fused = fusion(rgb_feats, depth_feats)
	out_s = decoder(fused[0], fused[1], fused[2], fused[3])

	l_reg = regression_loss(out_s, gt_s_norm)
	l_struct = structure_distillation_loss(out_s, mde_s)
	l_smooth = smoothness_loss(out_s, rgb_s)
	l_normal = normal_consistency_loss(out_s, gt_s_norm)
	total_loss = 1.0 * l_reg + 0.1 * l_smooth + 0.01 * l_struct + 0.1 * l_normal

	optimizer.zero_grad()
	total_loss.backward()
	optimizer.step()

	if (i + 1) % 10 == 0:
		print(f"iter {i+1}/{n_overfit_iters}, loss = {total_loss.item()}")
		print(l_reg, l_struct, l_smooth, l_normal)


# ---- final prediction, denormalize back to mm ----
with torch.no_grad():
	rgb_feats = rgb_encoder(rgb_s)
	depth_feats = depth_encoder(depth_s, mask_s)
	fused = fusion(rgb_feats, depth_feats)
	out_s = decoder(fused[0], fused[1], fused[2], fused[3])
	out_mm = (out_s + 1) / 2 * (depth_max - depth_min) + depth_min

pred_np = out_mm.squeeze().numpy()
gt_np = gt_s.squeeze().numpy()
error_np = abs(pred_np - gt_np)
mae = np.mean(np.abs(pred_np - gt_np))
mse = np.mean((pred_np - gt_np)**2)
rmse = np.sqrt(mse)
abs_rel = np.mean(np.abs(pred_np - gt_np) / gt_np)
print(f"MAE: {mae:.4f}  MSE: {mse:.4f}  RMSE: {rmse:.4f}  AbsRel: {abs_rel:.4f}")

frame_id = os.path.basename(dataset.rgb_files[8]).split('_rgb')[0]

print(f"Depth range - pred: {pred_np.min():.2f} to {pred_np.max():.2f}")
print(f"Error range - {error_np.min():.2f} to {error_np.max():.2f}")

plt.imsave(f"{frame_id}_pred_cconv.png", pred_np, cmap='viridis')
plt.imsave(f"{frame_id}_error_cconv.png", error_np, cmap='hot')

print(f"saved {frame_id}_pred_cconv.png and {frame_id}_error_cconv.png")

# ---- split error by whether original tof had valid data there ----
mask_np = mask_s.squeeze().numpy().astype(bool)  # True = tof had valid data

hole_error = np.abs(pred_np - gt_np)[~mask_np]
valid_error = np.abs(pred_np - gt_np)[mask_np]

print(f"HOLE regions   - MAE: {hole_error.mean():.4f}  count: {hole_error.size}")
print(f"VALID regions  - MAE: {valid_error.mean():.4f}  count: {valid_error.size}")

hole_absrel = (np.abs(pred_np - gt_np)[~mask_np] / gt_np[~mask_np]).mean()
valid_absrel = (np.abs(pred_np - gt_np)[mask_np] / gt_np[mask_np]).mean()
print(f"HOLE AbsRel: {hole_absrel:.4f}  VALID AbsRel: {valid_absrel:.4f}")
