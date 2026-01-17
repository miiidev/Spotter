import cv2
import os

# Paths
video_root = "data/c23"
frame_root = "data/frames"
fps_extract = 5  # Extract 5 frames per second

# Loop through videos
for root, dirs, files in os.walk(video_root):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            
            # Determine label path
            rel_path = os.path.relpath(root, video_root)
            out_dir = os.path.join(frame_root, rel_path, os.path.splitext(file)[0])
            os.makedirs(out_dir, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(int(video_fps / fps_extract), 1)
            
            frame_count = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if saved_count >= 128:
                    break
                if frame_count % frame_interval == 0:
                    frame_name = f"frame_{saved_count:04d}.jpg"
                    cv2.imwrite(os.path.join(out_dir, frame_name), frame)
                    saved_count += 1
                frame_count += 1
            
            cap.release()
            print(f"Saved {saved_count} frames from {file}")
