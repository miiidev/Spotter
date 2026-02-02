import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from dataset import VideoDataset
from model import TemporalCNN_GRU

DATA_ROOT = "data/processed"
MODEL_PATH = "checkpoints/temporal_model_best.pth"
BATCH_SIZE = 4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_split(split_dir):
    videos, labels = [], []
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        class_dir = os.path.join(split_dir, label_name)
        if not os.path.exists(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith(".mp4"):
                videos.append(os.path.join(class_dir, f))
                labels.append(label_id)
    return videos, labels

def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_dir = os.path.join(DATA_ROOT, "test")
    videos, labels = collect_split(test_dir)
    print(f"[INFO] Test videos: {len(videos)}")
    if len(videos) == 0:
        raise RuntimeError("❌ No test videos found")

    test_dataset = VideoDataset(videos, labels, split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = TemporalCNN_GRU(num_classes=2, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            outputs = model(videos)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["real (0)", "fake (1)"], digits=4)

    print("\n================ EVALUATION ================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("===========================================\n")

if __name__ == "__main__":
    main()
