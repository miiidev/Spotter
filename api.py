from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import shutil
import uuid
import os
import glob

from train.inference import predict_with_diagnostics

app = FastAPI(title="Spotter API")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Temp dir for processed videos
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "spotter_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SAMPLES_DIR = os.path.join("static", "samples")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/samples")
async def list_samples():
    """List available sample videos."""
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for f in sorted(os.listdir(SAMPLES_DIR)):
            if f.endswith((".mp4", ".avi", ".mov", ".webm")):
                # Derive a display name from the filename
                name = os.path.splitext(f)[0].replace("_", " ").title()
                samples.append({
                    "name": name,
                    "filename": f,
                    "url": f"/static/samples/{f}",
                })
    return {"samples": samples}


@app.post("/api/detect")
async def detect(video: UploadFile = File(...)):
    video_id = uuid.uuid4().hex
    original_filename = f"{video_id}_original.mp4"
    original_path = os.path.join(UPLOAD_DIR, original_filename)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    result = predict_with_diagnostics(original_path)

    diag_filename = f"{video_id}_diagnostic.mp4"
    diag_path = os.path.join(UPLOAD_DIR, diag_filename)
    shutil.move(result["diagnostic_video"], diag_path)

    return {
        "label": result["label"],
        "confidence": round(result["confidence"] * 100, 2),
        "original_url": f"/videos/{original_filename}",
        "diagnostic_url": f"/videos/{diag_filename}",
    }


@app.post("/api/detect-sample")
async def detect_sample(filename: str):
    """Run detection on a sample video."""
    sample_path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(sample_path):
        return {"error": "Sample not found"}

    # Copy sample to upload dir so it gets served the same way
    video_id = uuid.uuid4().hex
    original_filename = f"{video_id}_original.mp4"
    original_path = os.path.join(UPLOAD_DIR, original_filename)
    shutil.copy2(sample_path, original_path)

    result = predict_with_diagnostics(original_path)

    diag_filename = f"{video_id}_diagnostic.mp4"
    diag_path = os.path.join(UPLOAD_DIR, diag_filename)
    shutil.move(result["diagnostic_video"], diag_path)

    return {
        "label": result["label"],
        "confidence": round(result["confidence"] * 100, 2),
        "original_url": f"/videos/{original_filename}",
        "diagnostic_url": f"/videos/{diag_filename}",
    }


@app.get("/videos/{filename}")
async def serve_video(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    return {"error": "Not found"}