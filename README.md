<div align="center">

# 🔥 Spotter

**AI-Powered Deepfake Detection with Artifact Visualization**

![Spotter Banner](banner.png)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.5-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

Spotter is a deep learning system that detects deepfake videos using **EfficientNet-B0** and **Bidirectional GRU** temporal analysis, with **Grad-CAM heatmap** visualizations to highlight manipulated regions.

[Try It Live](#-try-it-live) · [Installation](#-installation) · [Usage](#-usage) · [Architecture](#-model-architecture) · [Training](#-training)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Try It Live](#-try-it-live)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
  - [Web Interface](#web-interface)
  - [Programmatic Inference](#programmatic-inference)
- [Model Architecture](#-model-architecture)
- [Detection Pipeline](#-detection-pipeline)
- [Training](#-training)
  - [Dataset Preparation](#1-dataset-preparation)
  - [Train the Model](#2-train-the-model)
  - [Evaluate](#3-evaluate)
- [Performance](#-performance)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Features

- 🎭 **Deepfake Detection** — Binary classification (Real vs. Fake) on video input
- 🔥 **Heatmap Visualization** — Pixelated Grad-CAM heatmaps highlight manipulation artifacts frame-by-frame
- 🧠 **Temporal Analysis** — Bidirectional GRU captures inconsistencies across video frames
- 👤 **Face-Focused** — MTCNN face detection isolates faces from background noise
- ⚡ **Fast Inference** — ~1–2 seconds per video with CUDA acceleration
- 🖥️ **Interactive UI** — Gradio-powered web interface for drag-and-drop detection

---

## 🚀 Try It Live

You can try Spotter on HuggingFace Spaces:

> 🔗 [**Launch on HuggingFace Spaces**](https://huggingface.co/spaces/miiidev/Spotter)

<!-- Replace the link above with your actual HuggingFace Space URL when deployed -->

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- FFmpeg (for video processing)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/miiidev/Spotter.git
cd Spotter

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model checkpoint
# Place the trained model at: checkpoints/spotter_v1.pth
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.10.0 | Deep learning framework |
| `efficientnet_pytorch` | 0.7.1 | EfficientNet-B0 backbone |
| `facenet-pytorch` | 2.5.3 | MTCNN face detection |
| `opencv-python` | 4.13.0 | Video/image processing |
| `gradio` | 6.5.1 | Web interface |
| `scipy` | 1.17.0 | Score interpolation |
| `numpy` | 2.4.2 | Numerical operations |

---

## 📁 Project Structure

```
Spotter/
├── app.py                    # Main Gradio web application
├── requirements.txt          # Python dependencies
├── checkpoints/              # Trained model weights
│   └── spotter_v1.pth        #   └── Production model checkpoint
├── scripts/
│   └── split_dataset.py      # Dataset splitting utility (train/val/test)
└── train/
    ├── model.py              # TemporalCNN_GRU model architecture
    ├── dataset.py            # Optimized VideoDataset with face caching
    ├── train.py              # Training loop with Focal Loss & early stopping
    ├── evaluate.py           # Model evaluation & metrics
    └── inference.py          # Inference engine with Grad-CAM & diagnostics
```

---

## 🎯 Usage

### Web Interface

Launch the interactive Gradio UI:

```bash
python app.py
```

The app will open at `http://localhost:7860`. Upload any video and click **🔍 Start Scan** to:

1. Get a **Real / Fake** prediction with confidence score
2. View a **diagnostic video** with Grad-CAM heatmaps overlaid on each frame

### Programmatic Inference

```python
from train.inference import predict_with_diagnostics

# Run detection on a video file
label, confidence, diagnostic_video_path = predict_with_diagnostics("path/to/video.mp4")

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}%")
print(f"Diagnostic video saved to: {diagnostic_video_path}")
```

---

## 🧠 Model Architecture

Spotter uses a two-stage architecture combining spatial feature extraction with temporal sequence modeling:

```
Video Input (24 frames @ 224×224)
        │
        ▼
┌─────────────────────┐
│   MTCNN Face Detector   │  ← Isolate face regions
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  EfficientNet-B0      │  ← Spatial features (1280-dim)
│  (ImageNet pretrained) │     Last 3 blocks fine-tuned
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Bidirectional GRU    │  ← Temporal modeling
│  (2 layers, 512 hidden)│     Output: 1024-dim (512×2)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Mean + Max Pooling   │  ← Temporal aggregation (2048-dim)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  MLP Classifier       │  ← 2048 → 512 → 128 → 2
│  (Dropout 0.5 / 0.3)  │
└─────────┬───────────┘
          │
          ▼
    Real / Fake + Grad-CAM Heatmap
```

### Key Components

| Component | Details |
|-----------|---------|
| **Backbone** | EfficientNet-B0 (1280-dim features, ImageNet pretrained) |
| **Temporal** | Bidirectional GRU (2 layers, 512 hidden units, 0.3 dropout) |
| **Pooling** | Mean + Max temporal pooling (2048-dim combined) |
| **Classifier** | 3-layer MLP with ReLU & Dropout |
| **Visualization** | Temporal Grad-CAM with pixelated heatmaps (6–16px grid) |

---

## 🔍 Detection Pipeline

```
1. Face Detection     →  MTCNN isolates faces from background
2. Frame Sampling     →  24 frames uniformly sampled from video
3. Feature Extraction →  EfficientNet-B0 encodes spatial features
4. Temporal Analysis  →  BiGRU captures cross-frame inconsistencies
5. Classification     →  Binary prediction (Real / Fake)
6. Visualization      →  Grad-CAM heatmaps highlight artifacts
```

### 🎨 What the Heatmap Shows

Red/warm pixelated regions indicate potential deepfake artifacts:

- 🔴 **Unnatural facial blending** — edges where the fake face meets the original
- 🔴 **Mouth/lip-sync inconsistencies** — mismatched lip movements
- 🔴 **Temporal artifacts** — flickering or jitter across frames
- 🟡 **Yellow box** — detected face bounding region

---

## 🏋️ Training

### 1. Dataset Preparation

Spotter is trained on [FaceForensics++](https://github.com/ondyari/FaceForensics). Organize your data as follows:

```
data/
├── real/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── fake/
    ├── video_001.mp4
    ├── video_002.mp4
    └── ...
```

Then split into train/val/test (70/15/15):

```bash
python scripts/split_dataset.py
```

This creates:

```
data/processed/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### 2. Train the Model

```bash
cd train
python train.py
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 8 |
| Epochs | 25 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight decay 1e-4) |
| Scheduler | Cosine Annealing with Warm Restarts (T₀=5) |
| Loss | Focal Loss (γ=2.0, class-weighted) |
| Early stopping | Patience = 7, min Δ = 0.5% |
| Mixed precision | ✅ AMP with GradScaler |
| Backbone freeze | Last 3 EfficientNet blocks + conv head unfrozen |

#### Dataset Optimizations

- **Cached face detection** — MTCNN runs once per video, results cached across epochs
- **Smart frame sampling** — Random start for training, uniform for validation
- **Data augmentation** — Horizontal flip, brightness/contrast, JPEG compression, rotation, blur, color jitter, Gaussian noise

### 3. Evaluate

```bash
cd train
python evaluate.py
```

Outputs accuracy, confusion matrix, and per-class classification report.

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 90% |
| **Real Video Recall** | 93% |
| **Fake Video Recall** | 86% |
| **Inference Time** | ~1–2 sec/video (GPU) |

> Evaluated on the FaceForensics++ test split.

---

## ⚙️ Technical Details

- **Input**: 24 frames per video, resized to 224×224, ImageNet-normalized
- **Face Detection**: MTCNN with 40px margin, confidence-based selection
- **Grad-CAM**: Temporal-aware, targets EfficientNet-B0's final conv layer
- **Heatmap Style**: Pixelated grid (6–16px), intensity = manipulation confidence
- **Score Aggregation**: Frame-level fake scores interpolated to full video duration
- **GPU Acceleration**: CUDA-accelerated when available, graceful CPU fallback

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Ideas for Contribution

- [ ] Support for additional deepfake detection datasets
- [ ] Multi-face detection in a single video
- [ ] Audio-visual deepfake detection
- [ ] Model distillation for mobile deployment
- [ ] REST API endpoint for batch processing

---

## 📝 Citation

If you use Spotter in your research, please cite:

```bibtex
@software{spotter2026,
  author       = {miiidev},
  title        = {Spotter: AI-Powered Deepfake Detection with Artifact Visualization},
  year         = {2026},
  url          = {https://github.com/miiidev/Spotter}
}
```

This project uses the **FaceForensics++** dataset for training and evaluation. If you use this work, please also cite:

```bibtex
@inproceedings{roessler2019faceforensicspp,
  author    = {Andreas R{\"o}ssler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
  title     = {FaceForensics++: Learning to Detect Manipulated Facial Images},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

---

## 📄 License

This project is open source. Please add a `LICENSE` file to specify terms.

---

<div align="center">

Made with 🔥 by the **Spotter Team** — Powered by EfficientNet-B0 + BiGRU

⭐ Star this repo if you found it useful!

</div>