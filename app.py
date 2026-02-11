import gradio as gr
import tempfile
import shutil
import os

from train.inference import predict_with_diagnostics

def run_detection(video):
    """
    Run deepfake detection on uploaded video.
    Returns prediction label, confidence score, and diagnostic video with heatmaps.
    """
    if video is None:
        return "⚠️ No video uploaded", 0.0, None

    try:
        # Run detection
        result = predict_with_diagnostics(video)

        return (
            f"🎭 {result['label']}",  # Add emoji for visual feedback
            round(result["confidence"] * 100, 2),  # Convert to percentage
            result["diagnostic_video"]
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0, None


# Custom CSS for better styling
custom_css = """
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    background: linear-gradient(90deg, #ff6b6b, #ee5a6f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5em;
}

#subtitle {
    text-align: center;
    font-size: 1.2em;
    color: #666;
    margin-bottom: 2em;
}

.gradio-container {
    max-width: 1200px !important;
}

#detect-btn {
    background: linear-gradient(90deg, #ff6b6b, #ee5a6f) !important;
    border: none !important;
    font-size: 1.1em !important;
    font-weight: bold !important;
}
"""

# Build the Gradio interface
with gr.Blocks(title="🔥 Spotter - Deepfake Detection", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("<h1 id='title'>🔥 HotSpot</h1>")
    gr.Markdown("<p id='subtitle'>AI-Powered Deepfake Detection with Artifact Visualization</p>")
    
    # Info box
    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown("""
        ### 🎯 Detection Process:
        1. **Face Detection**: Isolates faces from background noise
        2. **Temporal Analysis**: Analyzes frame sequences with BiGRU
        3. **Artifact Detection**: EfficientNet-B0 identifies manipulation patterns
        4. **Visualization**: Heatmaps highlight suspicious regions
        
        ### 📊 Performance:
        - **90% accuracy** on test dataset
        - **93% real video recall**
        - **86% fake video recall**
        
        ### 🎭 What the heatmap shows:
        Red pixelated regions indicate potential deepfake artifacts like:
        - Unnatural facial blending
        - Mouth/lip-sync inconsistencies  
        - Temporal artifacts across frames
        """)
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="📤 Upload Video",
                show_label=True
            )
            
            detect_btn = gr.Button(
                "🔍 Start Scan",
                variant="primary",
                size="lg",
                elem_id="detect-btn"
            )
            
            # Example videos (optional - add if you have samples)
            # gr.Examples(
            #     examples=[
            #         ["examples/real_sample.mp4"],
            #         ["examples/fake_sample.mp4"],
            #     ],
            #     inputs=video_input,
            #     label="Try example videos"
            # )
        
        with gr.Column(scale=1):
            video_output = gr.Video(
                label="🎥 Diagnostic Output (with heatmaps)",
                show_label=True
            )
    
    # Results section
    gr.Markdown("### 📋 Detection Results")
    with gr.Row():
        label_output = gr.Textbox(
            label="Prediction",
            placeholder="Results will appear here...",
            scale=2
        )
        confidence_output = gr.Number(
            label="Confidence (%)",
            precision=2,
            scale=1
        )
    
    # Footer info
    with gr.Accordion("⚙️ Technical Details", open=False):
        gr.Markdown("""
        ### 🧠 Model Architecture:
        - **Backbone**: EfficientNet-B0 (1280-dim features)
        - **Temporal**: Bidirectional GRU (2 layers, 512 hidden units)
        - **Input**: 24 frames @ 224x224 per video
        - **Output**: Binary classification (Real/Fake) + frame-level heatmaps
        
        ### 🎨 Visualization:
        - Pixelated heatmaps (6-16px grid) show artifact locations
        - Intensity correlates with manipulation confidence
        - Yellow box indicates detected face region
        
        ### 🚀 Performance:
        - **Inference time**: ~1-2 seconds per video
        - **GPU**: CUDA-accelerated (if available)
        - **Dataset**: Trained on FaceForensics++
        """)
    
    gr.Markdown("""
    ---
    <p style='text-align: center; color: #666;'>
    Made with 🔥 by HotSpot Team | Powered by EfficientNet-B0 + BiGRU
    </p>
    """)
    
    # Connect the button
    detect_btn.click(
        fn=run_detection,
        inputs=video_input,
        outputs=[label_output, confidence_output, video_output]
    )

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )