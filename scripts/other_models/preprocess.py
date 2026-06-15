# preprocess.py
import os, glob
import cv2, numpy as np

VIDEO_EXTS = (".mp4",".avi",".mov",".mkv")
NUM_FRAMES = 8
SIZE = 96

ROOT = "dataset/raw_videos"
SAVE = "dataset/processed"

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.resize(fr,(SIZE,SIZE))
        fr = cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
        fr = fr.astype(np.float32)/255.0
        frames.append(fr)
    cap.release()

    if len(frames) == 0:
        return np.zeros((3,NUM_FRAMES,SIZE,SIZE))

    idxs = np.linspace(0,len(frames)-1,NUM_FRAMES).astype(int)
    frames = [frames[i] for i in idxs]

    frames = np.stack(frames)              # (T,H,W,3)
    frames = np.transpose(frames,(3,0,1,2)) # (3,T,H,W)
    return frames

def main():
    classes = sorted(os.listdir(ROOT))

    for c in classes:
        os.makedirs(os.path.join(SAVE,c), exist_ok=True)

        for ext in VIDEO_EXTS:
            for p in glob.glob(os.path.join(ROOT,c,f"*{ext}")):
                frames = read_video(p)

                name = os.path.basename(p).split('.')[0]
                save_path = os.path.join(SAVE,c,name+".npy")

                np.save(save_path, frames)
                print("Saved:", save_path)

if __name__ == "__main__":
    main()