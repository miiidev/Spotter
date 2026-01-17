import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
INPUT_ROOT = "data/landmarks"
OUTPUT_ROOT = "data/FaceForensics++/processed"
MAX_FRAMES = 64
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# MediaPipe landmark indices
NOSE_IDX = 1
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =========================
# NORMALIZATION
# =========================
def normalize_landmarks(landmarks):
    landmarks = landmarks.copy()
    for t in range(landmarks.shape[0]):
        frame = landmarks[t]
        nose = frame[NOSE_IDX]
        frame -= nose
        left_eye = frame[LEFT_EYE_IDX]
        right_eye = frame[RIGHT_EYE_IDX]
        scale = np.linalg.norm(left_eye - right_eye) + 1e-6
        frame /= scale
        landmarks[t] = frame
    return landmarks


# =========================
# MOTION FEATURES
# =========================
def compute_motion_features(landmarks):
    velocity = landmarks[1:] - landmarks[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    return np.concatenate([velocity[1:], acceleration], axis=-1)


# =========================
# PAD / TRUNCATE
# =========================
def pad_or_truncate(sequence, max_len):
    T = sequence.shape[0]
    if T >= max_len:
        return sequence[:max_len]
    else:
        pad_shape = (max_len - T,) + sequence.shape[1:]
        pad = np.zeros(pad_shape, dtype=sequence.dtype)
        return np.concatenate([sequence, pad], axis=0)


# =========================
# PROCESS SINGLE FILE
# =========================
def process_file(input_path):
    data = np.load(input_path)
    if data.shape[0] < 5:
        return None  # too short / bad detection
    data = normalize_landmarks(data)
    data = compute_motion_features(data)
    data = pad_or_truncate(data, MAX_FRAMES)
    # Flatten to [T, 468*6]
    return data.reshape(MAX_FRAMES, -1)


# =========================
# MAIN PIPELINE
# =========================
for label in ["real", "fake"]:
    input_dir = os.path.join(INPUT_ROOT, label)
    all_files = []

    # Collect all files
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                all_files.append(os.path.join(root, file))

    # Split into train / val / test
    train_files, test_files = train_test_split(all_files, test_size=TEST_RATIO, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # Process each split
    for split_name, files_list in splits.items():
        out_dir = os.path.join(OUTPUT_ROOT, split_name, label)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nProcessing {label} - {split_name} ({len(files_list)} files)")

        for file_path in tqdm(files_list):
            data = process_file(file_path)
            if data is None:
                print(f"Skipped (too short): {file_path}")
                continue

            # Preserve subfolder structure
            rel_path = os.path.relpath(file_path, input_dir)
            out_subdir = os.path.join(out_dir, os.path.dirname(rel_path))
            os.makedirs(out_subdir, exist_ok=True)
            out_path = os.path.join(out_subdir, os.path.basename(file_path))
            np.save(out_path, data)
