import torch, os

processed_dir = "/content/data/scenario_1/processed"
files = sorted(os.listdir(processed_dir))
print(f"Files: {len(files)}")

# Check a sample file
sample = torch.load(os.path.join(processed_dir, files[0]), weights_only=True)
x = sample["x"]
y = sample["y"]
print(f"x shape: {x.shape}, dtype: {x.dtype}")
print(f"y shape: {y.shape}, dtype: {y.dtype}")
print(f"x action channels mean: {x[:, 16:].abs().mean():.6f}")
print(f"x action channels max: {x[:, 16:].abs().max():.6f}")
print(f"Click frames: {(x[:, 16:].abs().mean(dim=(1,2,3)) > 0.05).float().mean():.3f}")
