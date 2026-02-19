import gradio as gr
import json
import os

from train.inference import predict_with_diagnostics


def run_detection(video):
    """
    Run deepfake detection. Returns original video + diagnostic video.
    Both are passed to the custom HTML player.
    """
    if video is None:
        return "⚠️ No video uploaded", 0.0, "", gr.update(visible=False)

    try:
        result = predict_with_diagnostics(video)

        # Build the custom HTML player with both videos stacked
        html_player = build_toggle_player(
            result["original_video"],
            result["diagnostic_video"],
        )

        return (
            f"🎭 {result['label']}",
            round(result["confidence"] * 100, 2),
            html_player,
            gr.update(visible=True),
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0, "", gr.update(visible=False)


def build_toggle_player(original_path, diagnostic_path):
    """
    Build a custom HTML video player with two <video> elements stacked.
    JavaScript toggles visibility and syncs playback between them.
    The heatmap video sits on top, toggle just flips its opacity.
    """
    # Gradio serves files from its temp dir — use relative file paths
    original_url = f"/file={original_path}"
    diagnostic_url = f"/file={diagnostic_path}"

    return f"""
    <div id="spotter-player" style="position: relative; width: 100%; max-width: 720px; margin: 0 auto; border-radius: 12px; overflow: hidden; background: #000;">

        <!-- Original video (always plays, bottom layer) -->
        <video id="sp-original" src="{original_url}"
               style="width: 100%; display: block; border-radius: 12px;"
               controls playsinline>
        </video>

        <!-- Diagnostic video (top layer, toggled via opacity) -->
        <video id="sp-diagnostic" src="{diagnostic_url}"
               style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                      opacity: 1; pointer-events: none; border-radius: 12px;"
               playsinline muted>
        </video>

        <!-- Toggle button overlaid on video -->
        <button id="sp-toggle-btn" onclick="spotterToggle()"
                style="position: absolute; top: 12px; right: 12px; z-index: 10;
                       background: rgba(255,107,107,0.9); color: white; border: none;
                       padding: 8px 16px; border-radius: 8px; font-size: 14px;
                       font-weight: bold; cursor: pointer; backdrop-filter: blur(4px);
                       transition: background 0.2s;">
            🔥 Heatmap: ON
        </button>
    </div>

    <script>
    (function() {{
        const orig = document.getElementById('sp-original');
        const diag = document.getElementById('sp-diagnostic');
        const btn  = document.getElementById('sp-toggle-btn');
        let heatmapOn = true;

        // Sync diagnostic video to original video's time & play state
        function syncVideos() {{
            if (Math.abs(diag.currentTime - orig.currentTime) > 0.15) {{
                diag.currentTime = orig.currentTime;
            }}
        }}

        orig.addEventListener('play', () => {{
            diag.play();
            syncVideos();
        }});
        orig.addEventListener('pause', () => {{
            diag.pause();
            syncVideos();
        }});
        orig.addEventListener('seeked', () => {{
            syncVideos();
        }});
        orig.addEventListener('timeupdate', () => {{
            syncVideos();
        }});

        // Toggle function
        window.spotterToggle = function() {{
            heatmapOn = !heatmapOn;
            diag.style.opacity = heatmapOn ? '1' : '0';
            diag.style.transition = 'opacity 0.25s ease';
            btn.textContent = heatmapOn ? '🔥 Heatmap: ON' : '🔥 Heatmap: OFF';
            btn.style.background = heatmapOn
                ? 'rgba(255,107,107,0.9)'
                : 'rgba(100,100,100,0.9)';
        }};
    }})();
    </script>
    """


# Custom CSS
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
with gr.Blocks(
    title="🔥 Spotter - Deepfake Detection",
    theme=gr.themes.Soft(),
    css=custom_css,
) as demo:

    # Header
    gr.Markdown("<h1 id='title'>🔥 Spotter</h1>")
    gr.Markdown("<p id='subtitle'>AI-Powered Deepfake Detection with Artifact Visualization</p>")

    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown("""
        ### 🎯 Detection Process:
        1. **Face Detection**: Isolates faces from background noise
        2. **Temporal Analysis**: Analyzes frame sequences with BiGRU
        3. **Artifact Detection**: EfficientNet-B0 identifies manipulation patterns
        4. **Visualization**: Heatmaps highlight suspicious regions

        ### 🎭 What the heatmap shows:
        Red pixelated regions indicate potential deepfake artifacts like:
        - Unnatural facial blending
        - Mouth/lip-sync inconsistencies
        - Temporal artifacts across frames
        """)

    # Upload area (shown before scan)
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="📤 Upload Video")
            detect_btn = gr.Button(
                "🔍 Start Scan",
                variant="primary",
                size="lg",
                elem_id="detect-btn",
            )

    # Results area
    with gr.Row():
        label_output = gr.Textbox(label="🎭 Prediction", scale=1)
        confidence_output = gr.Number(label="📊 Confidence %", scale=1)

    # Custom HTML player (shown after scan)
    player_html = gr.HTML(value="", visible=False)

    # Scan button
    detect_btn.click(
        fn=run_detection,
        inputs=video_input,
        outputs=[
            label_output,
            confidence_output,
            player_html,
            player_html,       # visibility update
        ],
    )


if __name__ == "__main__":
    demo.launch()