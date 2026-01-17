import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FaceForensicsDataset(Dataset):
    """
    Dataset for FaceForensics++ landmarks.
    Expects folder structure:
    
    processed/
        train/
            real/
            fake/
        val/
            real/
            fake/
        test/
            real/
            fake/
    """
    def __init__(self, root_dir, split="train"):
        """
        root_dir: path to 'processed' folder
        split: one of 'train', 'val', 'test'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.samples = []

        for label_name, label_value in [("real", 0), ("fake", 1)]:
            folder = os.path.join(self.root_dir, label_name)
            if not os.path.exists(folder):
                continue

            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith(".npy"):
                        path = os.path.join(root, file)
                        self.samples.append((path, label_value))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.root_dir}")

        # Shuffle samples to mix real/fake
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path).astype(np.float32)  # [T, 468*6]
        # Convert to torch tensor
        data = torch.from_numpy(data)            # [T, F]
        label = torch.tensor(label, dtype=torch.float32)
        return data, label
