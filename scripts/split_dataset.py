import os
import shutil
import random

# ================= CONFIG =================
SOURCE_DIR = "data"
DEST_DIR = "data/processed"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
# ==========================================

random.seed(SEED)

def collect_videos(base_dir):
    videos = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                videos.append(os.path.join(root, f))
    return videos


def split_list(items):
    random.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:]
    }


for label in ["real", "fake"]:
    print(f"\n📂 Processing label: {label}")

    src_label_dir = os.path.join(SOURCE_DIR, label)
    videos = collect_videos(src_label_dir)

    if len(videos) == 0:
        raise RuntimeError(f"No videos found in {src_label_dir}")

    splits = split_list(videos)

    for split, vids in splits.items():
        out_dir = os.path.join(DEST_DIR, split, label)
        os.makedirs(out_dir, exist_ok=True)

        for v in vids:
            dst = os.path.join(out_dir, os.path.basename(v))
            shutil.copy2(v, dst)

        print(f"  {split}: {len(vids)} videos")

print("\n✅ Video dataset split completed successfully.")
