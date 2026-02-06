import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from dataset import VideoDataset
from model import TemporalCNN_GRU
import platform
import numpy as np

# =========================
# CONFIG
# =========================
DATA_ROOT = "data/processed"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "temporal_model_last.pth")
BATCH_SIZE = 8  # Reduced from 16 for more stable training
EPOCHS = 25  # Increased from 15
LR = 1e-4  # Reduced from 3e-4 for more stable learning
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_COMPILE = True  # torch.compile (disabled on Windows automatically)

# Early stopping config
PATIENCE = 7  # Stop if no improvement for 7 epochs
MIN_DELTA = 0.5  # Minimum improvement threshold (0.5%)

# =========================
# UTILS
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

# Focal Loss with better defaults
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for videos, labels in pbar:
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(videos)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, split_name="Val"):
    model.eval()
    correct = 0
    total = 0
    
    # Track per-class metrics
    true_positives = {0: 0, 1: 0}
    false_positives = {0: 0, 1: 0}
    false_negatives = {0: 0, 1: 0}
    
    pbar = tqdm(loader, desc=f"{split_name}", leave=False)
    for videos, labels in pbar:
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        
        with autocast():
            outputs = model(videos)
        
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Calculate per-class metrics
        for pred, label in zip(preds, labels):
            pred_item = pred.item()
            label_item = label.item()
            
            if pred_item == label_item:
                true_positives[label_item] += 1
            else:
                false_positives[pred_item] += 1
                false_negatives[label_item] += 1
        
        pbar.set_postfix({'acc': f'{100.0 * correct / total:.2f}%'})
    
    accuracy = 100.0 * correct / total
    
    # Calculate precision and recall for each class
    metrics = {}
    for class_id in [0, 1]:
        tp = true_positives[class_id]
        fp = false_positives[class_id]
        fn = false_negatives[class_id]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_name = "Real" if class_id == 0 else "Fake"
        metrics[class_name] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    return accuracy, metrics

# =========================
# MAIN
# =========================
def main():
    print("="*60)
    print("🚀 DEEPFAKE DETECTION TRAINING - IMPROVED PIPELINE")
    print("="*60)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print("="*60)

    # Load splits
    train_videos, train_labels = collect_split(os.path.join(DATA_ROOT, "train"))
    val_videos, val_labels = collect_split(os.path.join(DATA_ROOT, "val"))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Val videos: {len(val_videos)}")
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    print(f"  Train - Real: {train_counts[0]}, Fake: {train_counts[1]}")
    print(f"  Val - Real: {val_counts[0]}, Fake: {val_counts[1]}")
    
    if len(train_videos) == 0 or len(val_videos) == 0:
        raise RuntimeError("❌ Empty train or val split")

    # Datasets & loaders
    print("\n🔄 Creating datasets with face detection...")
    train_dataset = VideoDataset(
        train_videos, train_labels, 
        split="train", 
        augment=True, 
        size=224,  # Increased from 160 for better face details
        num_frames=24  # Reduced from 32 for speed (Phase 1 optimization!)
    )
    val_dataset = VideoDataset(
        val_videos, val_labels, 
        split="val", 
        augment=False, 
        size=224, 
        num_frames=24  # Reduced from 32 for speed
    )
    
    # PHASE 1 OPTIMIZATION: Warm up face detection cache!
    print("\n🔥 Phase 1 Optimization: Warming up face cache...")
    from dataset import warmup_face_cache
    warmup_face_cache(train_dataset)
    warmup_face_cache(val_dataset)
    print("✅ Cache ready! Training will be 3-5x faster now!\n")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        drop_last=True  # For more stable batch norm
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    # Model
    print("\n🧠 Initializing model...")
    model = TemporalCNN_GRU(
        num_classes=2, 
        hidden_size=512, 
        freeze_backbone=True  # Start with frozen backbone
    ).to(DEVICE)
    
    if USE_COMPILE and hasattr(torch, "compile") and platform.system() != "Windows":
        model = torch.compile(model)
        print("✅ Model compiled with torch.compile()")
    else:
        print("⚠️  torch.compile() skipped")

    # Loss with balanced class weights
    print("\n⚖️  Setting up loss function...")
    counts = Counter(train_labels)
    total = len(train_labels)
    weights = torch.tensor([
        total / (2.0 * counts[0]),  # Weight for class 0 (real)
        total / (2.0 * counts[1])   # Weight for class 1 (fake)
    ], device=DEVICE)
    print(f"  Class weights: Real={weights[0]:.3f}, Fake={weights[1]:.3f}")
    
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR,
        weight_decay=1e-4  # L2 regularization
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Double the restart period each time
        eta_min=1e-6
    )
    
    scaler = GradScaler()

    # Resume checkpoint if exists
    start_epoch = 1
    best_acc = 0.0
    best_f1_avg = 0.0
    epochs_without_improvement = 0
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n🔄 Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            best_f1_avg = checkpoint.get('best_f1_avg', 0.0)
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            print(f"  Resumed from epoch {checkpoint['epoch']}")
            print(f"  Best val acc: {best_acc:.2f}%")
            print(f"  Best F1 avg: {best_f1_avg:.2f}%")
        else:
            print("  ⚠️  Checkpoint format not recognized, starting fresh")

    # Unfreeze schedule: unfreeze backbone after epoch 10
    UNFREEZE_EPOCH = 10

    # Training loop
    print("\n" + "="*60)
    print("🎯 STARTING TRAINING")
    print("="*60)
    
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"📅 Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Unfreeze backbone after certain epoch
        if epoch == UNFREEZE_EPOCH:
            print("\n🔓 Unfreezing backbone for fine-tuning...")
            for param in model.feature_extractor.parameters():
                param.requires_grad = True
            # Reduce learning rate when unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * 0.1
            print(f"  Learning rate reduced to {LR * 0.1:.2e}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch
        )
        
        # Validate
        val_acc, val_metrics = evaluate(model, val_loader, split_name="Val")
        
        # Calculate average F1
        avg_f1 = (val_metrics['Real']['f1'] + val_metrics['Fake']['f1']) / 2.0
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"\n  Per-Class Metrics:")
        for class_name, metrics in val_metrics.items():
            print(f"    {class_name}:")
            print(f"      Precision: {metrics['precision']:.2f}%")
            print(f"      Recall: {metrics['recall']:.2f}%")
            print(f"      F1: {metrics['f1']:.2f}%")
        print(f"  Average F1: {avg_f1:.2f}%")

        # Check for improvement (use average F1 instead of just accuracy)
        improved = False
        if avg_f1 > best_f1_avg + MIN_DELTA:
            best_f1_avg = avg_f1
            best_acc = val_acc
            epochs_without_improvement = 0
            improved = True
            
            # Save best model
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "best_f1_avg": best_f1_avg,
                "val_metrics": val_metrics,
                "epochs_without_improvement": epochs_without_improvement
            }, os.path.join(CHECKPOINT_DIR, "temporal_model_best.pth"))
            print(f"\n✅ Saved best model (F1: {best_f1_avg:.2f}%, Acc: {best_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"\n⏳ No improvement for {epochs_without_improvement} epoch(s)")

        # Save last checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "best_f1_avg": best_f1_avg,
            "val_metrics": val_metrics,
            "epochs_without_improvement": epochs_without_improvement
        }, CHECKPOINT_PATH)
        print("💾 Saved checkpoint")
        
        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"\n🛑 Early stopping triggered! No improvement for {PATIENCE} epochs.")
            print(f"Best F1: {best_f1_avg:.2f}%, Best Acc: {best_acc:.2f}%")
            break

    print(f"\n{'='*60}")
    print("✅ TRAINING FINISHED")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Best Average F1 Score: {best_f1_avg:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()