import os
import cv2
import numpy as np
import mediapipe as mp

# Paths
frame_root = "data/frames"
landmark_root = "data/landmarks"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Loop through all frame folders
for root, dirs, files in os.walk(frame_root):
    if not files:
        continue  # Skip empty folders

    # Sort frames
    files.sort()
    
    landmarks_video = []

    for file in files:
        frame_path = os.path.join(root, file)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks_video.append(landmarks)
        else:
            # No face detected: append zeros
            landmarks_video.append(np.zeros((468, 3)))

    landmarks_video = np.array(landmarks_video)  # [num_frames, 468, 3]

    # Save as .npy
    rel_path = os.path.relpath(root, frame_root)
    out_dir = os.path.join(landmark_root, rel_path)
    os.makedirs(out_dir, exist_ok=True)
    video_name = os.path.basename(root)
    np.save(os.path.join(out_dir, f"{video_name}.npy"), landmarks_video)

    print(f"Saved landmarks for {video_name}, shape: {landmarks_video.shape}")
