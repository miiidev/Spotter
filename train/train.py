# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from dataset import VideoDataset
from model import TemporalCNN_GRU

# =========================
# CONFIG
# =========================
DATA_ROOT = "data/processed"
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# UTIL
# =========================
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

# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0

    for videos, labels in tqdm(loader, desc="Training", leave=False):
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Mixed precision forward/backward
        with autocast():
            outputs = model(videos)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for videos, labels in tqdm(loader, desc="Validating", leave=False):
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        with torch.cuda.amp.autocast():
            outputs = model(videos)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

# =========================
# MAIN
# =========================
def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # Load splits
    train_videos, train_labels = collect_split(os.path.join(DATA_ROOT, "train"))
    val_videos, val_labels = collect_split(os.path.join(DATA_ROOT, "val"))
    print(f"[INFO] Train videos: {len(train_videos)} | Val videos: {len(val_videos)}")

    if len(train_videos) == 0 or len(val_videos) == 0:
        raise RuntimeError("❌ Empty train or val split")

    # Datasets & loaders
    train_dataset = VideoDataset(train_videos, train_labels, split="train")
    val_dataset = VideoDataset(val_videos, val_labels, split="val")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = TemporalCNN_GRU(num_classes=NUM_WORKERS, freeze_backbone=True).to(DEVICE)

    # Class weights for imbalance
    counts = Counter(train_labels)
    weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # GradScaler for mixed precision
    scaler = GradScaler()

    # =========================
    # Training loop
    # =========================
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_acc = evaluate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/temporal_model_best.pth")
            print("✅ Saved best model")

    print(f"\nTraining finished. Best Val Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
