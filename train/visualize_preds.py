import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import FaceForensicsDataset
from model import DeepfakeLSTM

# =========================
# CONFIG
# =========================
DATA_ROOT = "data/FaceForensics++/processed"
MODEL_PATH = "checkpoints/best_model.pt"
BATCH_SIZE = 1         # visualize one video at a time
NUM_SAMPLES = 5        # number of test videos to plot
MAX_FRAMES = 64

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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# =========================
# MODEL
# =========================
model = DeepfakeLSTM(input_dim=468*6).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================
# VISUALIZATION FUNCTION
# =========================
def plot_prediction(sequence, label, prob, video_idx):
    """
    sequence: [T, F] flattened landmarks (used just for title)
    label: ground truth (0=real, 1=fake)
    prob: scalar probability (fake=1)
    """
    plt.figure(figsize=(6, 4))
    plt.bar([0,1], [1-prob, prob], color=['green','red'])
    plt.xticks([0,1], ['Real', 'Fake'])
    plt.ylim(0,1)
    plt.title(f"Video {video_idx} | Ground Truth: {'fake' if label==1 else 'real'}\nPredicted prob(fake)={prob:.3f}")
    plt.ylabel("Probability")
    plt.show()

# =========================
# RUN VISUALIZATION
# =========================
count = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()  # scalar per video

        for i in range(len(probs)):
            plot_prediction(
                sequence=x[i].cpu().numpy(),
                label=int(y[i].item()),
                prob=probs[i],
                video_idx=count+1
            )
            count += 1
            if count >= NUM_SAMPLES:
                break
        if count >= NUM_SAMPLES:
            break

