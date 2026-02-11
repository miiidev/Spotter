# 🔥 Spotter - AI-Powered Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/miiidev/Spotter?style=social)](https://github.com/miiidev/Spotter)

> A machine-learning tool that detects deepfake artifacts in videos with visual artifact analysis and heatmap visualization.

## 🎯 Overview

**Spotter** is an intelligent deepfake detection system that analyzes videos to spot synthetic face manipulations. Using advanced deep learning techniques, it provides:

- **High Accuracy**: 90% detection accuracy on test datasets
- **Interpretable Results**: Visual heatmaps showing suspicious regions
- **Fast Processing**: Real-time analysis with frame-by-frame diagnostics
- **User-Friendly Interface**: Web-based UI powered by Gradio

## ✨ Key Features

- 🎭 **Deepfake Detection**: Identifies synthetic face manipulations with high confidence
- 🔍 **Artifact Visualization**: Highlights suspicious regions with heatmaps
- 📊 **Comprehensive Analysis**: 
  - Face detection and isolation
  - Temporal frame analysis with BiGRU
  - EfficientNet-B0 based pattern detection
- 🌐 **Web Interface**: Interactive Gradio-based UI for easy video upload and analysis
- 📈 **Performance Metrics**:
  - 93% real video recall
  - 86% fake video detection rate

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- At least 4GB RAM (8GB recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/miiidev/Spotter.git
   cd Spotter