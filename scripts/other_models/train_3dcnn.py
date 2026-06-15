"""
train_3dcnn_fixed.py  —  3D CNN on raw videos
Fixes:
  - Split bug fixed (proper 80/10/10 with remainder handled)
  - ImageNet mean/std normalization
  - Data augmentation (random horizontal flip, random crop)
  - Added learning rate scheduler (CosineAnnealingLR)
  - Evaluation uses a proper held-out test split
  - Reduced SIZE to 96 and BATCH to 4 to be memory-safe
"""

import os, glob, json
import cv2, numpy as np, torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

NUM_FRAMES = 16
SIZE       = 96      # reduced from 112 for memory safety
BATCH      = 4       # reduced for CPU safety; use 8–16 on GPU
EPOCHS     = 60

# ImageNet stats for normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, root, classes, augment=False):
        self.augment = augment
        self.samples = []
        for i, c in enumerate(classes):
            for ext in VIDEO_EXTS:
                for p in glob.glob(os.path.join(root, c, f"*{ext}")):
                    self.samples.append((p, i))

    def __len__(self):
        return len(self.samples)

    def read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.resize(fr, (SIZE + 8, SIZE + 8))   # slightly larger for random crop
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(fr)
        cap.release()

        if len(frames) == 0:
            return np.zeros((3, NUM_FRAMES, SIZE, SIZE), dtype=np.float32)

        idxs = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
        frames = [frames[i] for i in idxs]
        frames = np.stack(frames).astype(np.float32) / 255.0   # (T,H,W,3)

        # ImageNet normalize
        frames = (frames - MEAN) / STD                          # broadcast (T,H,W,3)

        # Augmentation: random crop
        if self.augment:
            h0 = np.random.randint(0, 8 + 1)
            w0 = np.random.randint(0, 8 + 1)
        else:
            h0, w0 = 4, 4   # center crop
        frames = frames[:, h0:h0 + SIZE, w0:w0 + SIZE, :]

        # Augmentation: horizontal flip
        if self.augment and np.random.rand() < 0.5:
            frames = frames[:, :, ::-1, :]

        frames = np.transpose(frames, (3, 0, 1, 2))            # (C,T,H,W)
        return frames.copy().astype(np.float32)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        x = self.read_video(p)
        return torch.tensor(x), torch.tensor(l)


# ──────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────
class CNN3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),

            nn.Dropout3d(0.3)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


# ──────────────────────────────────────────────
#  Train
# ──────────────────────────────────────────────
def train():
    root    = "dataset/raw_videos"
    classes = sorted(os.listdir(root))
    print(f"Classes ({len(classes)}): {classes}")

    full_ds = VideoDataset(root, classes, augment=False)
    n       = len(full_ds)

    # ── Fixed split: 80 / 10 / 10 ──
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val          # absorbs remainder (was the bug)

    train_ds_base, val_ds_base, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Re-wrap train split with augmentation
    aug_ds  = VideoDataset(root, classes, augment=True)
    train_ds = torch.utils.data.Subset(aug_ds, train_ds_base.indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds_base, BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, BATCH, shuffle=False, num_workers=0)

    model = CNN3D(len(classes)).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ce    = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val  = 0.0
    best_path = "cnn3d_best.pth"

    for ep in range(EPOCHS):
        # ── Train loop ──
        model.train()
        correct = total = 0
        running_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            print(f"[3D] Epoch {ep+1}/{EPOCHS}  Batch {i+1}/{len(train_loader)}", end="\r")
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out  = model(x)
            loss = ce(out, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            correct      += (out.argmax(1) == y).sum().item()
            total        += y.size(0)

        train_acc  = correct / total
        train_loss = running_loss / len(train_loader)

        # ── Val loop ──
        model.eval()
        correct = total = val_loss_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out  = model(x)
                val_loss_total += ce(out, y).item()
                correct        += (out.argmax(1) == y).sum().item()
                total          += y.size(0)

        val_acc  = correct / total
        val_loss = val_loss_total / len(val_loader)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"\n[3D] Epoch {ep+1:02d}: TrainAcc={train_acc:.4f}  ValAcc={val_acc:.4f}  "
              f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(),
                        "num_classes": len(classes),
                        "classes": classes}, best_path)
            print(f"  ✅ Best model saved (val_acc={best_val:.4f})")

        sched.step()

    # ── Save history & plots ──
    with open("cnn3d_history.json", "w") as f:
        json.dump(history, f, indent=4)

    for metric in ("acc", "loss"):
        plt.figure()
        plt.plot(history[f"train_{metric}"], label=f"Train {metric}")
        plt.plot(history[f"val_{metric}"],   label=f"Val {metric}")
        plt.xlabel("Epoch"); plt.ylabel(metric.capitalize())
        plt.title(f"CNN3D {metric.capitalize()}")
        plt.legend()
        plt.savefig(f"cnn3d_{metric}.png")
        plt.close()

    # ── Final evaluation on TEST set ──
    print("\nEvaluating on held-out test set...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            all_preds.extend(model(x).argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    np.save("cnn3d_confusion_matrix.npy", cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("CNN3D Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("cnn3d_confusion_matrix.png")
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    with open("cnn3d_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    with open("cnn3d_classification_report.txt", "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=classes))

    class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = {classes[i]: float(class_acc[i]) for i in range(len(classes))}
    with open("cnn3d_per_class_accuracy.json", "w") as f:
        json.dump(per_class_acc, f, indent=4)

    print("✅ Training complete. All results saved.")


if __name__ == "__main__":
    train()