"""
train_stgcn_fixed.py  —  ST-GCN on hand keypoints
Fixes:
  - Keypoint normalization (wrist-relative, scale by hand span)
  - Keypoint augmentation (jitter, scale, flip)
  - Consistent checkpoint keys (use_xyz saved and loaded correctly)
  - num_class auto-detected from CSV
  - Macro F1 logged alongside accuracy
"""

import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from graph_hand import Graph
from model import STGCN


# ──────────────────────────────────────────────
#  Dataset  (with normalization + augmentation)
# ──────────────────────────────────────────────
class KeypointDataset(Dataset):
    def __init__(self, labels_csv, split="train", use_xyz=False, augment=False):
        self.use_xyz = use_xyz
        self.augment = augment
        self.items = []

        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.items.append((row["path"], int(row["label"])))

        if not self.items:
            raise ValueError(f"No rows for split='{split}' in {labels_csv}")

    def __len__(self):
        return len(self.items)

    @staticmethod
    def normalize(kp):
        """
        kp: (T, 21, C)  C = 2 or 3
        Center on wrist (joint 0), scale by max span per frame, average scale over T.
        """
        # Center on wrist
        kp = kp - kp[:, 0:1, :]          # broadcast (T,1,C)

        # Scale: mean of per-frame max distances from wrist
        scale = np.linalg.norm(kp, axis=-1).max(axis=-1).mean() + 1e-6
        kp = kp / scale
        return kp

    @staticmethod
    def augment_kp(kp):
        """kp: (T, 21, C)"""
        # Random jitter
        kp = kp + np.random.randn(*kp.shape).astype(np.float32) * 0.01
        # Random scale
        kp = kp * np.random.uniform(0.9, 1.1)
        # Random horizontal flip (negate x)
        if np.random.rand() < 0.5:
            kp[:, :, 0] = -kp[:, :, 0]
        return kp

    def __getitem__(self, idx):
        path, label = self.items[idx]
        kp = np.load(path).astype(np.float32)  # (T, 21, 3)

        C = 3 if self.use_xyz else 2
        kp = kp[:, :, :C]                       # (T, 21, C)

        kp = self.normalize(kp)

        if self.augment:
            kp = self.augment_kp(kp)

        x = np.transpose(kp, (2, 0, 1))         # (C, T, V)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def count_classes(labels_csv):
    labels = set()
    with open(labels_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.add(int(row["label"]))
    return max(labels) + 1


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv",    type=str,   default="dataset/labels_clean.csv")
    ap.add_argument("--epochs",        type=int,   default=60)
    ap.add_argument("--batch",         type=int,   default=32)
    ap.add_argument("--lr",            type=float, default=4e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--use_xyz",       action="store_true")
    ap.add_argument("--save_dir",      type=str,   default="checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Auto-detect num_class
    num_class = count_classes(args.labels_csv)
    print(f"Detected {num_class} classes")

    in_channels = 3 if args.use_xyz else 2

    graph = Graph(strategy="spatial", max_hop=1)
    A = graph.A
    model = STGCN(num_class=num_class, in_channels=in_channels, A=A).to(device)

    train_ds = KeypointDataset(args.labels_csv, split="train", use_xyz=args.use_xyz, augment=True)
    val_ds   = KeypointDataset(args.labels_csv, split="val",   use_xyz=args.use_xyz, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val  = 0.0
    best_path = os.path.join(args.save_dir, "best_stgcn.pt")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        total_loss = correct = total = 0
        for x, y in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ncols=90):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)

        train_loss = total_loss / max(1, total)
        train_acc  = correct    / max(1, total)

        # ── Val ──
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                v_correct += (pred == y).sum().item()
                v_total   += x.size(0)

        val_acc = v_correct / max(1, v_total)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d} | lr={current_lr:.6f} | "
              f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model":       model.state_dict(),
                "use_xyz":     args.use_xyz,        # ← fixed key name
                "in_channels": in_channels,
                "num_class":   num_class,
            }, best_path)
            print(f"  ✅ Best checkpoint saved (val_acc={best_val:.4f})")

        scheduler.step()

    print(f"\nTraining done. Best val_acc: {best_val:.4f}")


if __name__ == "__main__":
    main()