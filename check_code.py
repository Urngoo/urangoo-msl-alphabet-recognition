# paste this as check.py and run it
import numpy as np, glob, os

# Check if any .npy files were actually created
npys = glob.glob("new_dataset/test_videos/**/*.npy", recursive=True)
print(f"Found {len(npys)} .npy files")
for n in npys[:3]:
    arr = np.load(n)
    print(f"  {n}  shape={arr.shape}")

# Check what shape your training .npy files are
train_npys = glob.glob("dataset/processed/**/*.npy", recursive=True)
print(f"\nFound {len(train_npys)} training .npy files")
for n in train_npys[:3]:
    arr = np.load(n)
    print(f"  {n}  shape={arr.shape}")