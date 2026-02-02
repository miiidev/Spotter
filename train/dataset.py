import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN  # face detector
import random

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=32, size=224, split="train", augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.size = size
        self.split = split
        self.augment = augment

        # face detector
        self.mtcnn = MTCNN(image_size=size, margin=0, keep_all=False)

    def __len__(self):
        return len(self.video_paths)

    def _read_frame(self, cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None

        # detect face
        face = self.mtcnn(frame)
        if face is not None:
            frame = face.permute(1,2,0).numpy() * 255
            frame = frame.astype(np.uint8)
        else:
            frame = cv2.resize(frame, (self.size, self.size))

        return frame

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            frames = np.zeros((self.num_frames, self.size, self.size, 3), dtype=np.uint8)
        else:
            if self.split == "train":
                start = np.random.randint(0, max(1, total_frames - self.num_frames))
                indices = np.arange(start, start + self.num_frames)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

            frames = []
            for i in indices:
                frame = self._read_frame(cap, i)
                if frame is None:
                    frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)

                # augmentations
                if self.augment and self.split=="train":
                    if random.random() < 0.5:
                        frame = cv2.flip(frame, 1)  # horizontal flip
                    if random.random() < 0.3:
                        # color jitter
                        frame = np.clip(frame * (0.8 + 0.4*random.random()), 0, 255)

                frames.append(frame)

            frames = np.stack(frames)

        cap.release()

        # Normalize + tensor
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]

        # ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN, device=frames.device).view(1,3,1,1)
        std  = torch.tensor(IMAGENET_STD, device=frames.device).view(1,3,1,1)
        frames = (frames - mean) / std

        return frames, torch.tensor(label, dtype=torch.long)
