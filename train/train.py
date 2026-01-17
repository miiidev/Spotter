import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from dataset import FaceForensicsDataset
from model import DeepfakeLSTM

def main():
    # =========================
    # CONFIG
    # =========================
    DATA_ROOT = "data/FaceForensics++/processed"
    BATCH_SIZE = 8          # safe for RTX 3050 (6GB)
    EPOCHS = 20
    LR = 1e-4
    NUM_WORKERS = 4


    # =========================
    # DEVICE
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))


    # =========================
    # DATA
    # =========================
    train_dataset = FaceForensicsDataset(root_dir="data/FaceForensics++/processed", split="train")
    val_dataset   = FaceForensicsDataset(root_dir="data/FaceForensics++/processed", split="val")
    test_dataset  = FaceForensicsDataset(root_dir="data/FaceForensics++/processed", split="test")


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


    # =========================
    # MODEL
    # =========================
    model = DeepfakeLSTM().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)


    # =========================
    # TRAIN / VAL LOOPS
    # =========================
    def train_one_epoch(model, loader):
        model.train()
        total_loss = 0

        for x, y in tqdm(loader, desc="Training", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)


    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_preds = []
        all_labels = []

        for x, y in tqdm(loader, desc="Validating", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc = accuracy_score(all_labels > 0.5, all_preds > 0.5)
        auc = roc_auc_score(all_labels, all_preds)
        print("Unique labels in val:", set(all_labels))

        return acc, auc


    # =========================
    # MAIN TRAINING
    # =========================
    best_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader)
        val_acc, val_auc = evaluate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print("✅ Saved best model")

    print("\nTraining complete.")
    print(f"Best Val AUC: {best_auc:.4f}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
