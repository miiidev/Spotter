# inference.py
import cv2
import torch
import numpy as np
from train.model import TemporalCNN_GRU
import gradio as gr
import tempfile
import os
import uuid

# ========================= CONFIG =========================
NUM_FRAMES = 24
IMG_SIZE = 224
MODEL_PATH = "checkpoints/temporal_model_best.pth"
FPS = 24

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= Load model =========================
model = TemporalCNN_GRU(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ========================= Frame extraction =========================
def extract_frames(video_path, num_frames=NUM_FRAMES, size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)

    small_frames, full_frames = [], []

    if not cap.isOpened():
        black = np.zeros((size, size, 3), dtype=np.uint8)
        return np.stack([black] * num_frames), np.stack([black] * num_frames)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((size, size, 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        full_frames.append(frame_rgb)
        small_frames.append(cv2.resize(frame_rgb, (size, size)))

    cap.release()
    return np.stack(small_frames), np.stack(full_frames)

# ------------------------- Aggregate frame scores -------------------------
def aggregate_frame_scores(fake_scores, threshold=0.5):
    """
    Returns overall video label and confidence based on frame-level fake scores.
    """
    mean_score = fake_scores.mean()
    if mean_score >= threshold:
        return "FAKE", float(mean_score)
    else:
        return "REAL", float(1 - mean_score)

# ========================= Spatial heatmaps =========================
def compute_spatial_heatmaps(frames, target_class=1):
    """
    Grad-CAM style heatmaps using feature maps from the CNN backbone.
    """
    heatmaps = []

    for frame in frames:
        x = torch.from_numpy(frame.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)

        # Forward pass through feature extractor
        feature_maps = model.feature_extractor(x)  # (1, C, H, W)
        feature_maps.retain_grad()  # keep gradients

        # Forward through classifier
        pooled = feature_maps.view(1, -1)
        logits = model.classifier(pooled)
        score = logits[0, target_class]

        # Backward to get gradients w.r.t feature maps
        model.zero_grad()
        score.backward(retain_graph=True)
        grads = feature_maps.grad[0]  # (C, H, W)

        # Weight each feature map by the mean gradient
        weights = grads.mean(dim=(1,2))  # (C,)
        cam = torch.zeros(feature_maps.shape[2:], device=DEVICE)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * feature_maps[0,i]

        # ReLU + normalize
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        heatmaps.append(cam.detach().cpu().numpy())

    return np.stack(heatmaps)


# ========================= Diagnostics =========================
def frame_level_diagnostics(video_path):
    small_frames, full_frames = extract_frames(video_path)

    frames_tensor = torch.from_numpy(
        small_frames.astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, temporal_out = model(frames_tensor, return_temporal=True)
        B, T, D = temporal_out.shape

        temporal_flat = temporal_out.view(B * T, D)
        logits = model.classifier(temporal_flat)
        probs = torch.softmax(logits, dim=1)

        fake_scores = probs[:, 1].view(T).cpu().numpy()

    heatmaps = compute_spatial_heatmaps(small_frames)
    return full_frames, fake_scores, heatmaps

# ========================= Video writer =========================
def overlay_heatmap(frame_bgr, heatmap, alpha):
    heatmap = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_color, alpha, 0)

def generate_diagnostic_video(frames, fake_scores, heatmaps, out_path="diagnostic.mp4", fps=24):
    out_path = os.path.join(
        tempfile.gettempdir(),
        f"diagnostic_{uuid.uuid4().hex}.mp4"
    )

    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    for frame, score, heatmap in zip(frames, fake_scores, heatmaps):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        alpha = float(np.clip(score, 0.15, 0.8))
        frame_bgr = overlay_heatmap(frame_bgr, heatmap, alpha)

        cv2.putText(
            frame_bgr,
            f"Fake score: {score:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame_bgr)

    writer.release()
    return out_path

# ========================= Prediction =========================
def predict_with_diagnostics(video_path):
    frames, fake_scores, heatmaps = frame_level_diagnostics(video_path)

    # Use the aggregate function to get video-level label and confidence
    label, confidence = aggregate_frame_scores(fake_scores)

    diag_video = generate_diagnostic_video(
        frames, fake_scores, heatmaps, out_path="diagnostic.mp4"
    )

    return {
        "label": label,
        "confidence": confidence,
        "diagnostic_video": diag_video,
        "frame_scores": fake_scores
    }

# ========================= Gradio wrapper =========================
def gradio_predict(video):
    if video is None:
        return "No video uploaded", 0.0, None

    result = predict_with_diagnostics(video)
    return result["label"], result["confidence"], result["diagnostic_video"]

# ========================= UI =========================
if __name__ == "__main__":
    with gr.Blocks(title="Deepfake Detection") as demo:
        gr.Markdown("## 🎭 Deepfake Video Detection with Heatmaps")

        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            video_output = gr.Video(label="Diagnostic Output")

        with gr.Row():
            label_output = gr.Textbox(label="Prediction")
            confidence_output = gr.Number(label="Confidence")

        gr.Button("Run Detection").click(
            fn=gradio_predict,
            inputs=video_input,
            outputs=[label_output, confidence_output, video_output]
        )

    demo.launch()
