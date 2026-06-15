"""
train_temporal_pool_fixed.py  —  2D CNN + Temporal Pooling on raw videos
Fixes:
  - Split bug fixed (80/10/10 with remainder absorbed into test)
  - ImageNet normalization
  - Data augmentation (random crop, horizontal flip)
  - Replaced mean pooling with learned GRU temporal aggregation
    (so model is no longer order-invariant — critical for gestures)
  - Added CosineAnnealingLR scheduler
  - Evaluation uses proper held-out test set
  - Saved best model checkpoint
"""

import os, glob, json
import cv2, numpy as np, torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

NUM_FRAMES = 8
SIZE       = 64
BATCH      = 8
EPOCHS     = 60

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
            fr = cv2.resize(fr, (SIZE + 8, SIZE + 8))
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(fr)
        cap.release()

        if len(frames) == 0:
            return np.zeros((NUM_FRAMES, 3, SIZE, SIZE), dtype=np.float32)

        idxs   = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
        frames = [frames[i] for i in idxs]
        frames = np.stack(frames).astype(np.float32) / 255.0   # (T,H,W,3)
        frames = (frames - MEAN) / STD

        # Augmentation: random crop
        if self.augment:
            h0 = np.random.randint(0, 9)
            w0 = np.random.randint(0, 9)
        else:
            h0, w0 = 4, 4
        frames = frames[:, h0:h0 + SIZE, w0:w0 + SIZE, :]

        # Augmentation: horizontal flip
        if self.augment and np.random.rand() < 0.5:
            frames = frames[:, :, ::-1, :]

        frames = np.transpose(frames, (0, 3, 1, 2))            # (T,C,H,W)
        return frames.copy().astype(np.float32)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        x = self.read_video(p)
        return torch.tensor(x), torch.tensor(l)


# ──────────────────────────────────────────────
#  Model  (GRU replaces plain mean pooling)
# ──────────────────────────────────────────────
class CNNGRUPool(nn.Module):
    """
    Per-frame 2D CNN encoder → GRU across time → classify last hidden state.
    This preserves temporal ordering, unlike simple mean pooling.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.gru = nn.GRU(input_size=128, hidden_size=256,
                          num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Encode each frame
        x = x.view(B * T, C, H, W)
        f = self.enc(x).flatten(1)          # (B*T, 128)
        f = f.view(B, T, -1)               # (B, T, 128)
        # Temporal aggregation with GRU
        _, h = self.gru(f)                  # h: (1, B, 256)
        h = h.squeeze(0)                    # (B, 256)
        return self.fc(self.drop(h))


# ──────────────────────────────────────────────
#  Train
# ──────────────────────────────────────────────
def train():
    root    = "dataset/raw_videos"
    classes = sorted(os.listdir(root))
    print(f"Classes ({len(classes)}): {classes}")

    full_ds = VideoDataset(root, classes, augment=False)
    n       = len(full_ds)

    # Fixed split
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val

    base_train, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Re-wrap train with augmentation
    aug_ds   = VideoDataset(root, classes, augment=True)
    train_ds = torch.utils.data.Subset(aug_ds, base_train.indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=0)

    model = CNNGRUPool(len(classes)).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ce    = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    history   = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val  = 0.0
    best_path = "cnn_pool_best.pth"

    for ep in range(EPOCHS):
        model.train()
        correct = total = 0
        train_loss_total = 0.0

        for i, (x, y) in enumerate(train_loader):
            print(f"[POOL] Epoch {ep+1}/{EPOCHS}  Batch {i+1}/{len(train_loader)}", end="\r")
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out  = model(x)
            loss = ce(out, y)
            loss.backward()
            opt.step()

            train_loss_total += loss.item()
            correct          += (out.argmax(1) == y).sum().item()
            total            += y.size(0)

        train_acc  = correct / total
        train_loss = train_loss_total / len(train_loader)

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

        print(f"\n[POOL] Epoch {ep+1:02d}: TrainAcc={train_acc:.4f}  ValAcc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(),
                        "num_classes": len(classes),
                        "classes": classes}, best_path)
            print(f"  ✅ Best model saved (val_acc={best_val:.4f})")

        sched.step()

    # ── Plots ──
    with open("cnn_pool_history.json", "w") as f:
        json.dump(history, f, indent=4)

    for metric in ("acc", "loss"):
        plt.figure()
        plt.plot(history[f"train_{metric}"], label=f"Train {metric}")
        plt.plot(history[f"val_{metric}"],   label=f"Val {metric}")
        plt.xlabel("Epoch"); plt.ylabel(metric.capitalize())
        plt.title(f"CNN+GRU Pool {metric.capitalize()}")
        plt.legend()
        plt.savefig(f"cnn_pool_{metric}.png")
        plt.close()

    # ── Test evaluation ──
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
    np.save("cnn_pool_confusion_matrix.npy", cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("CNN+GRU Pool Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("cnn_pool_confusion_matrix.png")
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    with open("cnn_pool_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    with open("cnn_pool_classification_report.txt", "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=classes))

    class_acc = cm.diagonal() / cm.sum(axis=1)
    with open("cnn_pool_per_class_accuracy.json", "w") as f:
        json.dump({classes[i]: float(class_acc[i]) for i in range(len(classes))}, f, indent=4)

    print("✅ Training complete. All results saved.")


if __name__ == "__main__":
    train()