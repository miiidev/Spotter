import gradio as gr
import tempfile
import shutil
import os

from train.inference import predict_with_diagnostics

def run_detection(video):
    if video is None:
        return "No video uploaded", 0.0, None

    # Gradio provides a temp video path
    result = predict_with_diagnostics(video)

    return (
        result["label"],
        result["confidence"],
        result["diagnostic_video"]
    )

with gr.Blocks(title="VerifAI – Deepfake Detection") as demo:
    gr.Markdown("## 🎭 VerifAI – Deepfake Video Detection")

    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        video_output = gr.Video(label="Diagnostic Output")

    with gr.Row():
        label_output = gr.Textbox(label="Prediction")
        confidence_output = gr.Number(label="Confidence")

    detect_btn = gr.Button("Run Detection")

    detect_btn.click(
        fn=run_detection,
        inputs=video_input,
        outputs=[label_output, confidence_output, video_output]
    )

demo.launch()
