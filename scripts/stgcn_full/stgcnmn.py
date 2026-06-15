"""
realtime_stgcn.py  —  Real-time ST-GCN inference using MediaPipe hand keypoints
Place this file next to graph_hand.py and model.py

Usage:
    python realtime_stgcn.py --ckpt checkpoints/best_stgcn.pt
    python realtime_stgcn.py --ckpt checkpoints/best_stgcn.pt --camera 1
    python realtime_stgcn.py --ckpt checkpoints/best_stgcn.pt --use_xyz

Controls:
    Q or ESC  →  quit
    C         →  clear the frame buffer (reset)
"""

import os, sys, argparse, collections
import cv2
import numpy as np
import torch

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe not installed. Run:  pip install mediapipe")
    sys.exit(1)

# ST-GCN imports — must be in same folder or on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from graph_hand import Graph
    from model import STGCN
except ImportError:
    print("ERROR: graph_hand.py / model.py not found next to this script.")
    sys.exit(1)


# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
BUFFER_SIZE   = 30     # frames to accumulate before predicting (match training T)
STEP_SIZE     = 5      # predict every N new frames (sliding window)
SMOOTH_WINDOW = 5      # majority-vote over last N predictions for stability


# ──────────────────────────────────────────────
#  MONGOLIAN CYRILLIC DISPLAY MAP
#  Maps phonetic label → (Cyrillic letter, PIL font needed for rendering)
#  OpenCV cannot render Cyrillic natively, so we use PIL as a fallback.
# ──────────────────────────────────────────────
LABEL_TO_CYRILLIC = {
    "A":    "А",
    "B":    "Б",
    "V":    "В",
    "G":    "Г",
    "D":    "Д",
    "YE":   "Е",
    "YO":   "Ё",
    "J":    "Ж",
    "Z":    "З",
    "I":    "И",
    "hI":   "Й",
    "K":    "К",
    "L":    "Л",
    "M":    "М",
    "N":    "Н",
    "O":    "О",
    "OU":   "Ө",
    "P":    "П",
    "R":    "Р",
    "S":    "С",
    "T":    "Т",
    "U":    "У",
    "Y":    "Ү",
    "H":    "Х",
    "F":    "Ф",
    "TS":   "Ц",
    "CH":   "Ч",
    "SH":   "Ш",
    "SHCH": "Щ",
    "Htemdeg":  "Ъ",
    "ERII":     "Ы",
    "Ztemdeg":  "Ь",
    "E":    "Э",
    "YU":   "Ю",
    "YA":   "Я",
}

# Try to load PIL for Cyrillic rendering (OpenCV can't handle UTF-8 text)
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    print("[WARN] Pillow not installed — Cyrillic letters will show as '?'.")
    print("       Run:  pip install Pillow")

# Font path candidates (edit if your system font lives elsewhere)
_FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

def _find_font_path():
    """Try fc-match first (Linux/macOS), then fall back to candidate list."""
    # 1. Ask fontconfig for a Cyrillic-capable font
    try:
        import subprocess
        out = subprocess.check_output(
            ["fc-match", "--format=%{file}", ":lang=ru:spacing=proportional"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        if out and os.path.exists(out):
            print(f"[FONT] Using system font: {out}")
            return out
    except Exception:
        pass
    # 2. Walk candidate list
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            print(f"[FONT] Using font: {path}")
            return path
    print("[FONT] No TTF font found — Cyrillic may not render correctly.")
    return None

def _load_pil_font(size):
    if not _PIL_AVAILABLE:
        return None
    path = _find_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception as e:
            print(f"[FONT] truetype load failed ({e}), using PIL default.")
    return ImageFont.load_default()

_PIL_FONT_LARGE = _load_pil_font(72) if _PIL_AVAILABLE else None
_PIL_FONT_SMALL = _load_pil_font(48) if _PIL_AVAILABLE else None


# ──────────────────────────────────────────────
#  CYRILLIC RENDERING HELPER
# ──────────────────────────────────────────────
def draw_cyrillic(frame, text, x, y, font, color_bgr):
    """
    Renders a Cyrillic string onto a BGR OpenCV frame using PIL.
    x, y  — top-left corner of the visible glyph.
    color_bgr — OpenCV BGR tuple, e.g. (80, 255, 0).
    """
    if not _PIL_AVAILABLE or not text:
        return

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    # Measure the true bounding box (may have non-zero origin offsets)
    dummy = Image.new("RGBA", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    # bbox = (left, top, right, bottom) — left/top can be negative or >0
    left, top, right, bottom = bbox
    tw = right  - left + 4
    th = bottom - top  + 4

    if tw <= 0 or th <= 0:
        return

    fh, fw = frame.shape[:2]
    if x >= fw or y >= fh:
        return

    # Draw at (-left+2, -top+2) so the visible glyph starts at (2,2) in canvas
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(canvas)
    draw.text((-left + 2, -top + 2), text, font=font, fill=(*color_rgb, 255))

    # Composite onto frame
    x2 = min(x + tw, fw)
    y2 = min(y + th, fh)
    cw = x2 - x
    ch = y2 - y
    if cw <= 0 or ch <= 0:
        return

    region    = frame[y:y2, x:x2]
    canvas_np = np.array(canvas.crop((0, 0, cw, ch)))
    alpha = canvas_np[:, :, 3:4] / 255.0
    rgb   = canvas_np[:, :, :3][:, :, ::-1]
    frame[y:y2, x:x2] = (region * (1 - alpha) + rgb * alpha).astype(np.uint8)


# ──────────────────────────────────────────────
#  KEYPOINT NORMALIZATION  (same as training)
# ──────────────────────────────────────────────
def normalize_keypoints(kp):
    """
    kp: (T, 21, C)
    Returns wrist-centered, scale-normalized keypoints.
    """
    kp = kp - kp[:, 0:1, :]
    scale = np.linalg.norm(kp, axis=-1).max(axis=-1).mean() + 1e-6
    return kp / scale


# ──────────────────────────────────────────────
#  LOAD MODEL
# ──────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    in_channels = ckpt["in_channels"]
    num_class   = ckpt["num_class"]
    use_xyz     = ckpt.get("use_xyz", False)
    classes     = ckpt.get("classes", [str(i) for i in range(num_class)])

    graph = Graph(strategy="spatial", max_hop=1)
    model = STGCN(num_class=num_class, in_channels=in_channels, A=graph.A).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded ST-GCN | {num_class} classes | in_channels={in_channels} | use_xyz={use_xyz}")
    print(f"Classes: {classes}")
    return model, classes, in_channels, use_xyz


# ──────────────────────────────────────────────
#  INFERENCE ON BUFFERED KEYPOINTS
# ──────────────────────────────────────────────
def predict(model, kp_buffer, in_channels, device):
    """
    kp_buffer: list of (21, 3) arrays, length = BUFFER_SIZE
    Returns (class_idx, confidence, probs_array)
    """
    kp = np.stack(kp_buffer, axis=0).astype(np.float32)   # (T, 21, 3)
    C  = in_channels
    kp = kp[:, :, :C]                                      # (T, 21, C)
    kp = normalize_keypoints(kp)
    x  = np.transpose(kp, (2, 0, 1))                       # (C, T, V)
    x  = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1,C,T,V)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    return pred_idx, float(probs[pred_idx]), probs


# ──────────────────────────────────────────────
#  DRAW HAND LANDMARKS ON FRAME
# ──────────────────────────────────────────────
def draw_landmarks(frame, hand_landmarks, color=(0, 255, 80)):
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands          = mp.solutions.hands

    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    required=True, help="Path to best_stgcn.pt")
    ap.add_argument("--camera",  type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--use_xyz", action="store_true",
                    help="Use xyz coords (must match training; auto-detected from ckpt)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, classes, in_channels, use_xyz = load_model(args.ckpt, device)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    kp_buffer        = []
    frame_since_pred = 0
    pred_label       = "—"
    pred_cyrillic    = ""
    pred_conf        = 0.0
    pred_history     = collections.deque(maxlen=SMOOTH_WINDOW)
    hand_detected    = False

    print(f"\n[LIVE] ST-GCN webcam | buffer={BUFFER_SIZE} frames | Press Q to quit, C to clear buffer\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_detected = result.multi_hand_landmarks is not None

        if hand_detected:
            lm = result.multi_hand_landmarks[0]
            draw_landmarks(frame, lm)

            kp = np.array([[l.x, l.y, l.z] for l in lm.landmark], dtype=np.float32)
            kp_buffer.append(kp)
            frame_since_pred += 1

            if len(kp_buffer) > BUFFER_SIZE:
                kp_buffer = kp_buffer[-BUFFER_SIZE:]

            if len(kp_buffer) == BUFFER_SIZE and frame_since_pred >= STEP_SIZE:
                frame_since_pred = 0
                idx, conf, probs = predict(model, kp_buffer, in_channels, device)
                pred_history.append(idx)

                smoothed_idx  = max(set(pred_history), key=list(pred_history).count)
                pred_label    = classes[smoothed_idx] if smoothed_idx < len(classes) else str(smoothed_idx)
                pred_cyrillic = LABEL_TO_CYRILLIC.get(pred_label, "")
                pred_conf     = conf
        else:
            if kp_buffer:
                kp_buffer.pop(0)

        # ── UI Overlay ──
        h, w = frame.shape[:2]

        # Top bar background — tall enough for the full glyph
        cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)

        # ── Cyrillic letter — y=5 gives top padding, bar is tall enough below ──
        display_text = pred_cyrillic if pred_cyrillic else pred_label
        draw_cyrillic(frame, display_text, 20, 5, _PIL_FONT_SMALL, (80, 255, 0))

        # Confidence percentage — vertically centred in bar
        conf_text = f"{pred_conf * 100:.1f}%"
        cv2.putText(frame, conf_text, (220, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 220, 0), 2)

        # Buffer fill bar sits just below the top bar
        buf_ratio = len(kp_buffer) / BUFFER_SIZE
        bar_w     = int(w * buf_ratio)
        cv2.rectangle(frame, (0, 98), (bar_w, 102), (0, 180, 255), -1)

        # Hand detection status
        status_color = (0, 255, 80) if hand_detected else (0, 0, 220)
        status_text  = "Hand detected" if hand_detected else "No hand detected"
        cv2.putText(frame, status_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Model name
        cv2.putText(frame, "ST-GCN | Mongolian Sign Language",
                    (w - 420, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Buffer count
        cv2.putText(frame, f"Buffer: {len(kp_buffer)}/{BUFFER_SIZE}",
                    (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("ST-GCN Real-Time Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("c"), ord("C")):
            kp_buffer.clear()
            pred_history.clear()
            pred_label    = "—"
            pred_cyrillic = ""
            pred_conf     = 0.0
            print("[INFO] Buffer cleared.")

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()