"""
extract_keypoints.py  —  Extract hand keypoints from videos using MediaPipe
Places a .npy file next to each video file.

Usage:
    python extract_keypoints.py --source new_dataset/test_videos/
    python extract_keypoints.py --source new_dataset/test_videos/A/A_001.mp4

Install dependency first:
    pip install mediapipe
"""

import os, sys, glob, argparse
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe not installed. Run:  pip install mediapipe")
    sys.exit(1)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
NUM_FRAMES = 30   # number of keypoint frames to sample per video


def extract_keypoints_from_video(video_path, num_frames=NUM_FRAMES):
    """
    Returns: np.ndarray of shape (T, 21, 3)  — x, y, z per joint
             or None if no hand was detected in any frame
    """
    mp_hands   = mp.solutions.hands
    cap        = cv2.VideoCapture(video_path)
    raw_frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) == 0:
        print(f"  [WARN] Could not read frames: {video_path}")
        return None

    # Sample evenly
    idxs   = np.linspace(0, len(raw_frames) - 1, num_frames).astype(int)
    frames = [raw_frames[i] for i in idxs]

    keypoints = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    ) as hands:
        for fr in frames:
            rgb    = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm  = result.multi_hand_landmarks[0].landmark
                kp  = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # (21,3)
            else:
                # No hand detected — use zeros (will be filled by interpolation below)
                kp = None

            keypoints.append(kp)

    # Interpolate missing frames
    # Find frames where detection succeeded
    valid_indices = [i for i, kp in enumerate(keypoints) if kp is not None]

    if len(valid_indices) == 0:
        print(f"  [WARN] No hand detected in any frame: {video_path}")
        return None

    # Fill missing frames with nearest valid keypoint
    filled = []
    for i, kp in enumerate(keypoints):
        if kp is not None:
            filled.append(kp)
        else:
            # Find nearest valid frame
            nearest = min(valid_indices, key=lambda v: abs(v - i))
            filled.append(keypoints[nearest])

    result_array = np.stack(filled, axis=0)  # (T, 21, 3)
    return result_array


def process_path(source):
    # Collect all video files
    if os.path.isfile(source):
        video_files = [source]
    elif os.path.isdir(source):
        video_files = []
        for ext in VIDEO_EXTS:
            video_files.extend(glob.glob(os.path.join(source, "**", f"*{ext}"), recursive=True))
        video_files.sort()
    else:
        print(f"Source not found: {source}")
        sys.exit(1)

    if not video_files:
        print(f"No video files found in: {source}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s). Extracting keypoints...\n")

    success = 0
    failed  = []

    for vp in video_files:
        npy_path = os.path.splitext(vp)[0] + ".npy"

        if os.path.exists(npy_path):
            print(f"  [SKIP] Already exists: {npy_path}")
            success += 1
            continue

        print(f"  Processing: {vp}")
        kp = extract_keypoints_from_video(vp)

        if kp is None:
            print(f"  [FAIL] Skipped (no hand detected): {vp}")
            failed.append(vp)
            continue

        np.save(npy_path, kp)
        print(f"  [OK]   Saved: {npy_path}  shape={kp.shape}")
        success += 1

    print(f"\n✅ Done: {success}/{len(video_files)} extracted successfully.")
    if failed:
        print(f"⚠️  {len(failed)} file(s) failed (no hand detected):")
        for f in failed:
            print(f"    {f}")
        print("\nTip: These files may have poor lighting or the hand is out of frame.")
        print("     ST-GCN will skip them during testing.")


def main():
    global NUM_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Video file or folder (e.g. new_dataset/test_videos/)")
    ap.add_argument("--num_frames", type=int, default=NUM_FRAMES,
                    help=f"Keypoint frames to extract per video (default: {NUM_FRAMES})")
    args = ap.parse_args()

    NUM_FRAMES = args.num_frames

    process_path(args.source)


if __name__ == "__main__":
    main()