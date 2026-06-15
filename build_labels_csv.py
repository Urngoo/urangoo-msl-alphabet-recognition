"""
build_labels_csv.py  —  Scan extracted .npy keypoint files and build labels_clean.csv
Run this AFTER extract_keypoints.py has processed your raw_videos folder.

Usage:
    python build_labels_csv.py --source dataset/raw_videos/ --out dataset/labels_clean.csv
"""

import os, sys, glob, argparse, csv, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="dataset/raw_videos/",
                    help="Root folder containing class subfolders with .npy files")
    ap.add_argument("--out", default="dataset/labels_clean.csv",
                    help="Output CSV path")
    ap.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    ap.add_argument("--val",   type=float, default=0.1, help="Val split ratio")
    ap.add_argument("--seed",  type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    classes = sorted(os.listdir(args.source))
    classes = [c for c in classes if os.path.isdir(os.path.join(args.source, c))]
    print(f"Found {len(classes)} classes: {classes}")

    rows = []
    for label_idx, cls in enumerate(classes):
        npy_files = glob.glob(os.path.join(args.source, cls, "*.npy"))
        if not npy_files:
            print(f"  [WARN] No .npy files found for class '{cls}' — run extract_keypoints.py first")
            continue
        for npy in npy_files:
            rows.append({
                "path":  os.path.abspath(npy).replace("\\", "/"),
                "label": label_idx,
                "class": cls,
                "split": None   # assigned below
            })

    if not rows:
        print("No .npy files found at all. Run extract_keypoints.py on your raw_videos folder first.")
        sys.exit(1)

    # Shuffle and assign splits per class (stratified)
    final_rows = []
    for label_idx, cls in enumerate(classes):
        cls_rows = [r for r in rows if r["class"] == cls]
        random.shuffle(cls_rows)
        n       = len(cls_rows)
        n_train = max(1, int(n * args.train))
        n_val   = max(1, int(n * args.val))
        # Remaining goes to test
        for i, r in enumerate(cls_rows):
            if i < n_train:
                r["split"] = "train"
            elif i < n_train + n_val:
                r["split"] = "val"
            else:
                r["split"] = "test"
        final_rows.extend(cls_rows)

    # Save CSV
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "class", "split"])
        writer.writeheader()
        writer.writerows(final_rows)

    # Summary
    train_n = sum(1 for r in final_rows if r["split"] == "train")
    val_n   = sum(1 for r in final_rows if r["split"] == "val")
    test_n  = sum(1 for r in final_rows if r["split"] == "test")

    print(f"\nSaved: {args.out}")
    print(f"  Total : {len(final_rows)}")
    print(f"  Train : {train_n}")
    print(f"  Val   : {val_n}")
    print(f"  Test  : {test_n}")
    print(f"\nClass mapping:")
    for i, c in enumerate(classes):
        print(f"  {i:2d} → {c}")

if __name__ == "__main__":
    main()