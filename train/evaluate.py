import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

from dataset import FaceForensicsDataset
from model import DeepfakeLSTM

def main():
    # =========================
    # CONFIG
    # =========================
    DATA_ROOT = "data/FaceForensics++/processed"
    BATCH_SIZE = 8
    MODEL_PATH = "checkpoints/best_model.pt"

    # =========================
    # DEVICE
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # =========================
    # DATASET
    # =========================
    test_dataset = FaceForensicsDataset(DATA_ROOT, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,        # <- fix for Windows multiprocessing
        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================
    model = DeepfakeLSTM(input_dim=468*6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # =========================
    # RUN INFERENCE
    # =========================
    all_labels = []
    all_logits = []

    with torch.no_grad():
        total_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            all_labels.append(y.cpu())
            all_logits.append(logits.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs > 0.5).astype(int)

    # =========================
    # METRICS
    # =========================
    avg_loss = total_loss / len(test_dataset)
    accuracy = accuracy_score(all_labels, preds)
    cm = confusion_matrix(all_labels, preds)
    report = classification_report(all_labels, preds, target_names=["real (0)", "fake (1)"])

    print("_________________________________________________________")
    print(f"Loss:     {avg_loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("_________________________________________________________")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
