# inference_efficientnet.py
import cv2
import torch
import numpy as np
from train.model import TemporalCNN_GRU
import tempfile
import os
import uuid
from facenet_pytorch import MTCNN
from scipy.interpolate import interp1d

# ========================= CONFIG =========================
NUM_FRAMES = 24
IMG_SIZE = 224
MODEL_PATH = "checkpoints/spotter_v1.pth"
FPS = 24

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= Load model =========================
model = TemporalCNN_GRU(num_classes=2)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
# Handle both direct state_dict and training checkpoint formats
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    model.load_state_dict(checkpoint)
    print("✅ Loaded model weights")
model.to(DEVICE)
model.eval()

# ========================= Select target layer for Grad-CAM =========================
# For EfficientNet, use the final conv head for best spatial resolution
target_layer = model.get_feature_extractor_layer('final')
print(f"✅ Using Grad-CAM layer: {target_layer.__class__.__name__}")

# ========================= Face Detection Setup =========================
# Initialize MTCNN for face detection
face_detector = MTCNN(
    image_size=IMG_SIZE,
    margin=40,
    keep_all=False,
    device=DEVICE,
    post_process=False
)

def detect_and_crop_face(frame_rgb):
    """
    Detect and crop face from frame.
    Returns cropped face or center crop as fallback.
    """
    try:
        boxes, probs = face_detector.detect(frame_rgb)
        
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].astype(int)
            x1, y1, x2, y2 = box
            
            # Add margin
            margin = 20
            h, w = frame_rgb.shape[:2]
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            face = frame_rgb[y1:y2, x1:x2]
            if face.size > 0:
                return cv2.resize(face, (IMG_SIZE, IMG_SIZE)), (x1, y1, x2, y2)
    except:
        pass
    
    # Fallback: center crop
    h, w = frame_rgb.shape[:2]
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    center_crop = frame_rgb[start_h:start_h+crop_size, start_w:start_w+crop_size]
    
    if center_crop.size > 0:
        return cv2.resize(center_crop, (IMG_SIZE, IMG_SIZE)), None
    
    return cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE)), None


# ========================= Video metadata =========================
def get_video_info(video_path):
    """Get video metadata without reading frames."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


# ========================= Optimized frame extraction =========================
def extract_analyzed_frames(video_path, num_frames=NUM_FRAMES, size=IMG_SIZE):
    """
    Extract ONLY the frames needed for model analysis.
    Reads the video once, only decodes frames at analyzed indices.

    Returns:
        - small_frames: (num_frames, size, size, 3) - cropped faces for model input
        - analyzed_face_boxes: list of face boxes for analyzed frames
        - analyzed_indices: indices of frames that were analyzed
        - video_info: dict with total_frames, fps, width, height
    """
    info = get_video_info(video_path)
    total_frames = info["total_frames"]

    if total_frames == 0:
        black = np.zeros((size, size, 3), dtype=np.uint8)
        return (np.stack([black] * num_frames),
                [None] * num_frames,
                np.array([0]),
                info)

    # Calculate which frames to analyze
    analyzed_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    target_set = set(analyzed_indices.tolist())

    cap = cv2.VideoCapture(video_path)
    small_frames = []
    analyzed_face_boxes = []
    current_idx = 0
    next_target = 0  # pointer into analyzed_indices

    while next_target < len(analyzed_indices) and cap.isOpened():
        ret = cap.grab()  # grab() is faster — only decodes header
        if not ret:
            break

        if current_idx == analyzed_indices[next_target]:
            ret, frame = cap.retrieve()  # only fully decode frames we need
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_cropped, face_box = detect_and_crop_face(frame_rgb)
                small_frames.append(face_cropped)
                analyzed_face_boxes.append(face_box)
            next_target += 1

        current_idx += 1

    cap.release()

    # Handle edge case where we got fewer frames than expected
    while len(small_frames) < num_frames:
        small_frames.append(np.zeros((size, size, 3), dtype=np.uint8))
        analyzed_face_boxes.append(None)

    print(f"✅ Extracted {len(small_frames)} analyzed frames (skipped {total_frames - len(small_frames)})")

    return (np.stack(small_frames),
            analyzed_face_boxes,
            analyzed_indices,
            info)


# ========================= Temporal-Aware Grad-CAM =========================
class TemporalGradCAM:
    """
    Grad-CAM that hooks into CNN backbone while running the full temporal model.
    Optimized for EfficientNet backbone.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_heatmaps(self, video_tensor, target_class):
        """
        Generate per-frame heatmaps using the FULL temporal model.
        
        Args:
            video_tensor: (1, T, C, H, W) - batch of frames
            target_class: which class to compute CAM for (0=real, 1=fake)
        
        Returns:
            heatmaps: (T, H, W) - one heatmap per frame
            frame_scores: (T,) - per-frame fake probability
        """
        self.model.zero_grad()
        video_tensor.requires_grad = True
        
        # Forward pass through FULL model (CNN + GRU + classifier)
        logits, temporal_out = self.model(video_tensor, return_temporal=True)
        
        # Get the score for target class
        score = logits[0, target_class]
        
        # Backward to get gradients
        score.backward(retain_graph=True)
        
        # EfficientNet feature maps are larger than ResNet!
        # Activations shape: (B*T, C, H, W) where H,W are larger (e.g., 7x7 or higher)
        B, T, _ = temporal_out.shape
        num_channels = self.activations.shape[1]
        h, w = self.activations.shape[2:]
        
        print(f"[DEBUG] Grad-CAM resolution: {h}x{w} (EfficientNet advantage!)")
        
        # Reshape activations and gradients to separate batch and time
        activations = self.activations.view(B, T, num_channels, h, w)
        gradients = self.gradients.view(B, T, num_channels, h, w)
        
        heatmaps = []
        
        for t in range(T):
            # Get activations and gradients for this frame
            act = activations[0, t]      # (C, H, W)
            grad = gradients[0, t]       # (C, H, W)
            
            # Compute weights (global average pooling of gradients)
            weights = grad.mean(dim=(1, 2))  # (C,)
            
            # Weighted combination of activation maps
            cam = torch.zeros((h, w), device=DEVICE)
            for i in range(num_channels):
                cam += weights[i] * act[i]
            
            # ReLU and normalize
            cam = torch.relu(cam)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            heatmaps.append(cam.cpu().numpy())
        
        # Also compute per-frame fake scores using temporal features
        with torch.no_grad():
            fake_scores_list = []
            
            for t in range(T):
                # Take temporal output at this timestep (D,)
                frame_temporal = temporal_out[0, t, :]
                
                # Create pooled representation by concatenating the same features
                pooled_frame = torch.cat([frame_temporal, frame_temporal]).unsqueeze(0)
                
                frame_logits = self.model.classifier(pooled_frame)
                frame_probs = torch.softmax(frame_logits, dim=1)
                fake_scores_list.append(frame_probs[0, 1].item())
            
            fake_scores = np.array(fake_scores_list)
        
        return np.stack(heatmaps), fake_scores

# Initialize Grad-CAM
gradcam = TemporalGradCAM(model, target_layer)

# ========================= Interpolation functions =========================
def interpolate_scores(fake_scores, analyzed_indices, total_frames):
    """
    Interpolate fake scores for all frames using linear interpolation.
    """
    if len(analyzed_indices) == 1:
        return np.full(total_frames, fake_scores[0])
    
    f = interp1d(analyzed_indices, fake_scores, kind='linear', fill_value='extrapolate')
    all_frame_indices = np.arange(total_frames)
    interpolated_scores = f(all_frame_indices)
    interpolated_scores = np.clip(interpolated_scores, 0, 1)
    
    return interpolated_scores


def interpolate_heatmaps(heatmaps, analyzed_indices, total_frames):
    """
    Interpolate heatmaps for all frames using nearest-neighbor.
    """
    h, w = heatmaps[0].shape
    all_heatmaps = np.zeros((total_frames, h, w))
    
    for frame_idx in range(total_frames):
        nearest_pos = np.argmin(np.abs(analyzed_indices - frame_idx))
        all_heatmaps[frame_idx] = heatmaps[nearest_pos]
    
    return all_heatmaps

# ========================= Aggregate frame scores -------------------------
def aggregate_frame_scores(fake_scores, threshold=0.5):
    """
    Returns overall video label and confidence based on frame-level fake scores.
    """
    mean_score = fake_scores.mean()
    if mean_score >= threshold:
        return "FAKE", float(mean_score)
    else:
        return "REAL", float(1 - mean_score)

# ========================= Optimized diagnostics =========================
def analyze_video(video_path):
    """
    Step 1: Extract only the frames needed for model analysis,
    run the model, and return lightweight analysis results.
    Does NOT generate any video yet.

    Returns:
        analysis dict with: label, confidence, fake_scores,
        heatmaps, analyzed_indices, video_info
    """
    small_frames, analyzed_face_boxes, analyzed_indices, video_info = \
        extract_analyzed_frames(video_path)

    # Prepare tensor (normalize with ImageNet stats)
    frames_tensor = torch.from_numpy(
        small_frames.astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)

    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 1, 3, 1, 1)
    frames_tensor = (frames_tensor - mean) / std

    # Predict class
    with torch.no_grad():
        logits = model(frames_tensor)
        pred_class = logits.argmax(dim=1).item()

    # Generate heatmaps for analyzed frames only
    analyzed_heatmaps, analyzed_fake_scores = gradcam.generate_heatmaps(
        frames_tensor, target_class=pred_class
    )

    # Interpolate to all frames
    total_frames = video_info["total_frames"]
    all_heatmaps = interpolate_heatmaps(analyzed_heatmaps, analyzed_indices, total_frames)
    all_fake_scores = interpolate_scores(analyzed_fake_scores, analyzed_indices, total_frames)

    label, confidence = aggregate_frame_scores(all_fake_scores)

    print(f"🎬 Analysis complete: {label} ({confidence:.1%}) — "
          f"analyzed {len(analyzed_indices)}/{total_frames} frames")

    return {
        "label": label,
        "confidence": confidence,
        "fake_scores": all_fake_scores,
        "heatmaps": all_heatmaps,
        "analyzed_indices": analyzed_indices,
        "video_info": video_info,
    }


# ========================= Video writer =========================
def create_pixelated_heatmap(heatmap, grid_size=12):
    """
    Create a pixelated/blocky heatmap - FINER grid for EfficientNet.
    """
    h, w = heatmap.shape
    pixelated = np.zeros_like(heatmap)
    
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            cell = heatmap[i:i+grid_size, j:j+grid_size]
            if cell.size > 0:
                cell_value = cell.max()
                pixelated[i:i+grid_size, j:j+grid_size] = cell_value
    
    return pixelated

def overlay_heatmap(frame_bgr, heatmap, alpha, threshold=0.3, grid_size=12):
    """
    Overlay pixelated heatmap on frame with threshold for artifact highlighting.
    """
    heatmap_resized = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))
    pixelated_heatmap = create_pixelated_heatmap(heatmap_resized, grid_size=grid_size)
    mask = pixelated_heatmap > threshold
    
    heatmap_colored = np.zeros_like(frame_bgr)
    heatmap_colored[:, :, 2] = (pixelated_heatmap * 255).astype(np.uint8)
    heatmap_colored[:, :, 0] = ((1 - pixelated_heatmap) * 100).astype(np.uint8)
    
    result = frame_bgr.copy()
    result[mask] = cv2.addWeighted(
        frame_bgr[mask], 1 - alpha,
        heatmap_colored[mask], alpha, 0
    )
    
    overlay_grid = result.copy()
    for i in range(0, frame_bgr.shape[0], grid_size):
        if mask[min(i, mask.shape[0]-1), :].any():
            cv2.line(overlay_grid, (0, i), (frame_bgr.shape[1], i), (255, 255, 255), 1)
    for j in range(0, frame_bgr.shape[1], grid_size):
        if mask[:, min(j, mask.shape[1]-1)].any():
            cv2.line(overlay_grid, (j, 0), (j, frame_bgr.shape[0]), (255, 255, 255), 1)
    
    result = cv2.addWeighted(result, 0.85, overlay_grid, 0.15, 0)
    return result


def generate_diagnostic_video_lazy(video_path, fake_scores, heatmaps, fps=FPS):
    """
    Generate diagnostic video by re-reading the original video lazily.
    Processes one frame at a time — never stores all frames in RAM.
    Uses face tracking to reduce MTCNN calls.
    """
    out_path = os.path.join(
        tempfile.gettempdir(),
        f"diagnostic_{uuid.uuid4().hex}.mp4"
    )

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        video_fps,
        (w, h)
    )

    print(f"🎬 Writing diagnostic video: {total_frames} frames at {video_fps:.0f} FPS...")

    # Face tracking: detect every N frames, reuse box between detections
    face_detect_interval = max(1, int(video_fps // 6))  # ~6 detections per second
    cached_face_box = None

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_bgr = frame  # already BGR from cap.read()

        # Face detection with tracking (only run MTCNN periodically)
        if frame_idx % face_detect_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            _, cached_face_box = detect_and_crop_face(frame_rgb)

        score = fake_scores[frame_idx] if frame_idx < len(fake_scores) else 0.0
        heatmap = heatmaps[frame_idx] if frame_idx < len(heatmaps) else np.zeros((7, 7))
        face_box = cached_face_box

        # Overlay heatmap on face region or full frame
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            face_w, face_h = x2 - x1, y2 - y1

            if face_w > 0 and face_h > 0:
                heatmap_resized = cv2.resize(heatmap, (face_w, face_h))
                face_region = frame_bgr[y1:y2, x1:x2].copy()

                alpha = float(np.clip(score * 0.7, 0.2, 0.6))
                grid_size = max(6, min(16, face_h // 18))

                face_region = overlay_heatmap(
                    face_region, heatmap_resized, alpha,
                    threshold=0.3, grid_size=grid_size
                )
                frame_bgr[y1:y2, x1:x2] = face_region
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        else:
            alpha = float(np.clip(score * 0.7, 0.2, 0.6))
            grid_size = max(8, min(24, h // 20))
            frame_bgr = overlay_heatmap(
                frame_bgr, heatmap, alpha,
                threshold=0.3, grid_size=grid_size
            )

        # Info overlay
        overlay_bg = frame_bgr.copy()
        cv2.rectangle(overlay_bg, (5, 5), (380, 110), (0, 0, 0), -1)
        frame_bgr = cv2.addWeighted(frame_bgr, 0.7, overlay_bg, 0.3, 0)

        cv2.putText(frame_bgr, f"Frame: {frame_idx + 1}/{total_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_bgr, f"Fake Confidence: {score:.1%}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255) if score < 0.5 else (0, 100, 255), 2)

        face_status = "Face Detected ✓" if face_box is not None else "No Face (Fallback)"
        cv2.putText(frame_bgr, face_status, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if face_box is not None else (0, 165, 255), 2)

        if face_box is not None and (x2 - x1) > 0 and (y2 - y1) > 0:
            hm = cv2.resize(heatmap, (x2 - x1, y2 - y1))
        else:
            hm = cv2.resize(heatmap, (w, h))
        artifact_pct = ((hm > 0.3).sum() / hm.size) * 100
        cv2.putText(frame_bgr, f"Artifact Regions: {artifact_pct:.1f}%", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame_bgr)

        if (frame_idx + 1) % max(1, total_frames // 10) == 0:
            print(f"   Processing frame {frame_idx + 1}/{total_frames}...")

    cap.release()
    writer.release()
    print(f"✅ Diagnostic video saved to {out_path}")
    return out_path


# ========================= Main prediction entry point =========================
def predict_with_diagnostics(video_path):
    """
    Run detection with optimized lazy diagnostic video generation.
    Returns both the original video path and the diagnostic video path.
    """
    # Step 1: Analyze only 24 frames (fast, low memory)
    analysis = analyze_video(video_path)

    # Step 2: Lazily generate diagnostic video (re-reads original, 1 frame at a time)
    diag_video = generate_diagnostic_video_lazy(
        video_path,
        analysis["fake_scores"],
        analysis["heatmaps"],
    )

    return {
        "label": analysis["label"],
        "confidence": analysis["confidence"],
        "diagnostic_video": diag_video,
        "original_video": video_path,
        "frame_scores": analysis["fake_scores"],
    }