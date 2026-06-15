"""
realtime_test.py  —  Real-time inference on all three models
Usage:
    python realtime_test.py --model stgcn   --ckpt checkpoints/best_stgcn.pt
    python realtime_test.py --model cnn3d   --ckpt cnn3d_best.pth
    python realtime_test.py --model cnnpool --ckpt cnn_pool_best.pth

Test on a file (matches your path format  new_dataset/test_video/A/A_001.mp4):
    python realtime_test.py --model stgcn --ckpt checkpoints/best_stgcn.pt \
                            --source new_dataset/test_video/A/A_001.mp4

Test a whole folder of videos (prints per-file predictions):
    python realtime_test.py --model cnn3d --ckpt cnn3d_best.pth \
                            --source new_dataset/test_video/

Live webcam:
    python realtime_test.py --model stgcn --ckpt checkpoints/best_stgcn.pt \
                            --source 0

Dependencies that must be importable:
    graph_hand.py  model.py  (only needed for --model stgcn)
"""

import os, sys, glob, argparse, time
import cv2, numpy as np, torch
import torch.nn as nn

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# ── ImageNet stats (used by 3D CNN and Pool models) ──
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (self-contained so no extra imports
#  are needed for 3D CNN / Pool)
# ════════════════════════════════════════════════════════════

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


class CNNGRUPool(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.gru  = nn.GRU(128, 256, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.enc(x.view(B * T, C, H, W)).flatten(1).view(B, T, -1)
        _, h = self.gru(f)
        return self.fc(self.drop(h.squeeze(0)))


# ════════════════════════════════════════════════════════════
#  PREPROCESSING HELPERS
# ════════════════════════════════════════════════════════════

def preprocess_video_3dcnn(frames_bgr, num_frames=16, size=96):
    """frames_bgr: list of BGR np arrays → (1,3,T,H,W) tensor"""
    if len(frames_bgr) == 0:
        return torch.zeros(1, 3, num_frames, size, size)
    idxs   = np.linspace(0, len(frames_bgr) - 1, num_frames).astype(int)
    frames = [frames_bgr[i] for i in idxs]
    out    = []
    for fr in frames:
        fr = cv2.resize(fr, (size, size))
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        fr = (fr - MEAN) / STD
        out.append(fr)
    out = np.stack(out)                     # (T,H,W,3)
    out = np.transpose(out, (3, 0, 1, 2))  # (3,T,H,W)
    return torch.tensor(out, dtype=torch.float32).unsqueeze(0)  # (1,3,T,H,W)


def preprocess_video_pool(frames_bgr, num_frames=8, size=64):
    """→ (1,T,3,H,W) tensor"""
    if len(frames_bgr) == 0:
        return torch.zeros(1, num_frames, 3, size, size)
    idxs   = np.linspace(0, len(frames_bgr) - 1, num_frames).astype(int)
    frames = [frames_bgr[i] for i in idxs]
    out    = []
    for fr in frames:
        fr = cv2.resize(fr, (size, size))
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        fr = (fr - MEAN) / STD
        out.append(np.transpose(fr, (2, 0, 1)))   # (3,H,W)
    out = np.stack(out)                             # (T,3,H,W)
    return torch.tensor(out, dtype=torch.float32).unsqueeze(0) # noqa


def preprocess_keypoints(kp_npy, use_xyz=False):
    """
    kp_npy: (T, 21, 3)  → (1, C, T, V) tensor
    Applies same wrist-relative normalization as training.
    """
    kp = kp_npy.astype(np.float32)
    C  = 3 if use_xyz else 2
    kp = kp[:, :, :C]
    kp = kp - kp[:, 0:1, :]                           # wrist-relative
    scale = np.linalg.norm(kp, axis=-1).max(axis=-1).mean() + 1e-6
    kp = kp / scale
    x  = np.transpose(kp, (2, 0, 1))                  # (C, T, V)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)   # (1,C,T,V)


# ════════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════════

def load_model(model_type, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if model_type == "stgcn":
        # ST-GCN needs graph_hand.py and model.py
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from stgcn_full.graph_hand import Graph
            from stgcn_full.model import STGCN
        except ImportError:
            print("ERROR: graph_hand.py / model.py not found next to this script.")
            sys.exit(1)

        in_channels = ckpt["in_channels"]
        num_class   = ckpt["num_class"]
        use_xyz     = ckpt.get("use_xyz", False)

        graph = Graph(strategy="spatial", max_hop=1)
        model = STGCN(num_class=num_class, in_channels=in_channels, A=graph.A).to(device)
        model.load_state_dict(ckpt["model"])
        classes = ckpt.get("classes", [str(i) for i in range(num_class)])
        return model, classes, use_xyz

    elif model_type == "cnn3d":
        num_classes = ckpt.get("num_classes", ckpt.get("num_class"))
        classes     = ckpt.get("classes", [str(i) for i in range(num_classes)])
        model       = CNN3D(num_classes).to(device)
        model.load_state_dict(ckpt["model"])
        return model, classes, None

    elif model_type == "cnnpool":
        num_classes = ckpt.get("num_classes", ckpt.get("num_class"))
        classes     = ckpt.get("classes", [str(i) for i in range(num_classes)])
        model       = CNNGRUPool(num_classes).to(device)
        model.load_state_dict(ckpt["model"])
        return model, classes, None

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ════════════════════════════════════════════════════════════
#  INFERENCE ON A SINGLE VIDEO FILE
# ════════════════════════════════════════════════════════════

def infer_video_file(video_path, model, model_type, classes, use_xyz, device):
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    model.eval()
    with torch.no_grad():
        if model_type == "stgcn":
            # For ST-GCN test from video, we need keypoints.
            # If a matching .npy exists next to the video, use it.
            npy_path = os.path.splitext(video_path)[0] + ".npy"
            if not os.path.exists(npy_path):
                print(f"  [STGCN] No keypoint .npy found at {npy_path}. "
                      "Run MediaPipe extraction first. Skipping.")
                return None
            kp  = np.load(npy_path)
            x   = preprocess_keypoints(kp, use_xyz).to(device)
            out = model(x)

        elif model_type == "cnn3d":
            x   = preprocess_video_3dcnn(frames).to(device)
            out = model(x)

        elif model_type == "cnnpool":
            x = preprocess_video_pool(frames).to(device)
            out = model(x)

        probs    = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_cls = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
        conf     = float(probs[pred_idx])

    return pred_cls, conf, probs


# ════════════════════════════════════════════════════════════
#  BATCH TEST ON A DIRECTORY  (new_dataset/test_video/A/A_001.mp4 …)
# ════════════════════════════════════════════════════════════

def test_directory(root, model, model_type, classes, use_xyz, device):
    video_files = []
    for ext in VIDEO_EXTS:
        video_files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    video_files.sort()

    if not video_files:
        print(f"No video files found in {root}")
        return

    correct = total = 0
    results = []

    for vp in video_files:
        # Ground truth = parent folder name (your structure: .../A/A_001.mp4 → label "A")
        true_label = os.path.basename(os.path.dirname(vp))
        result     = infer_video_file(vp, model, model_type, classes, use_xyz, device)
        if result is None:
            continue
        pred_cls, conf, _ = result

        is_correct = (pred_cls.upper() == true_label.upper())
        correct   += int(is_correct)
        total     += 1
        status     = "✅" if is_correct else "❌"
        print(f"  {status}  {os.path.relpath(vp, root):40s}  "
              f"true={true_label:4s}  pred={pred_cls:4s}  conf={conf:.3f}")
        results.append({"file": vp, "true": true_label, "pred": pred_cls, "conf": conf})

    print(f"\n── Test Accuracy: {correct}/{total} = {correct/max(1,total):.4f} ──")
    return results


# ════════════════════════════════════════════════════════════
#  LIVE WEBCAM / STREAM
# ════════════════════════════════════════════════════════════

def live_inference(source, model, model_type, classes, use_xyz, device,
                   buffer_size_3d=16, buffer_size_pool=8):
    """
    Sliding-window real-time inference.
    source: int (webcam index) or RTSP/file string
    For ST-GCN live mode we cannot run MediaPipe here easily,
    so we show a helpful message and fall back to CNN models.
    """
    if model_type == "stgcn":
        print("[STGCN] Live webcam mode requires real-time MediaPipe keypoint extraction.")
        print("  → Tip: pipe MediaPipe output to this script or use a pre-recorded .npy stream.")
        print("  → For quick live testing, use --model cnn3d or --model cnnpool instead.")
        return

    buffer_size = buffer_size_3d if model_type == "cnn3d" else buffer_size_pool
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)

    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    frame_buffer = []
    pred_label   = "—"
    pred_conf    = 0.0
    fps_timer    = time.time()
    frame_count  = 0

    print(f"[LIVE] Running {model_type.upper()} | Press Q to quit")
    model.eval()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_buffer.append(frame.copy())
        frame_count += 1

        # Run inference every `buffer_size` frames (sliding: drop oldest)
        if len(frame_buffer) >= buffer_size:
            with torch.no_grad():
                if model_type == "cnn3d":
                    x   = preprocess_video_3dcnn(frame_buffer[-buffer_size:]).to(device)
                    out = model(x)
                else:
                    raw = preprocess_video_pool(frame_buffer[-buffer_size:])
                    x   = torch.tensor(raw.numpy(), dtype=torch.float32).to(device)
                    out = model(x)

                probs      = torch.softmax(out, dim=1)[0].cpu().numpy()
                pred_idx   = int(probs.argmax())
                pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
                pred_conf  = float(probs[pred_idx])

            # Keep buffer rolling (overlap = buffer_size // 2)
            frame_buffer = frame_buffer[-(buffer_size // 2):]

        # ── Overlay on frame ──
        elapsed = time.time() - fps_timer
        fps = frame_count / max(elapsed, 1e-6)

        disp = frame.copy()
        h, w = disp.shape[:2]

        cv2.rectangle(disp, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(disp, f"Prediction: {pred_label}  ({pred_conf*100:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
        cv2.putText(disp, f"Model: {model_type.upper()}   FPS: {fps:.1f}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Real-Time Gesture Recognition", disp)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Real-time / batch test for gesture recognition models")
    ap.add_argument("--model",  required=True, choices=["stgcn", "cnn3d", "cnnpool"],
                    help="Which model to load")
    ap.add_argument("--ckpt",   required=True,
                    help="Path to checkpoint (.pt or .pth)")
    ap.add_argument("--source", default="0",
                    help="Video file, folder of videos, or webcam index (default: 0). "
                         "Example: new_dataset/test_video/A/A_001.mp4  "
                         "or  new_dataset/test_video/  (whole folder)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, classes, use_xyz = load_model(args.model, args.ckpt, device)
    model.eval()
    print(f"Loaded {args.model.upper()} | {len(classes)} classes: {classes}")

    src = args.source

    # ── Case 1: directory → batch test ──
    if os.path.isdir(src):
        print(f"\n📂 Batch testing folder: {src}\n")
        test_directory(src, model, args.model, classes, use_xyz, device)

    # ── Case 2: single video file ──
    elif os.path.isfile(src) and any(src.lower().endswith(e) for e in VIDEO_EXTS):
        print(f"\n🎬 Single file: {src}")
        result = infer_video_file(src, model, args.model, classes, use_xyz, device)
        if result:
            pred_cls, conf, probs = result
            print(f"\nPrediction : {pred_cls}")
            print(f"Confidence : {conf*100:.2f}%")
            print("\nTop-3:")
            top3 = np.argsort(probs)[::-1][:3]
            for i in top3:
                lbl = classes[i] if i < len(classes) else str(i)
                print(f"  {lbl:15s}: {probs[i]*100:.2f}%")

    # ── Case 3: webcam / stream ──
    else:
        live_inference(src, model, args.model, classes, use_xyz, device)


if __name__ == "__main__":
    main()