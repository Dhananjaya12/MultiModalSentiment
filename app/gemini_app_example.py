"""
Standalone Gemini multimodal sentiment demo.

This file is intentionally separate from the existing project inference path.
It does not import the trained TransformerFusionModel, does not load local
checkpoints, and does not modify the current Gradio app.

How to run on Kaggle/Colab:

    !pip install -q google-genai gradio

    import os
    os.environ["GEMINI_API_KEY"] = "YOUR_KEY_HERE"

    !python app/gemini_app_example.py

For Kaggle secrets:

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")

    !python app/gemini_app_example.py
"""

from __future__ import annotations

import html
import mimetypes
import os
from pathlib import Path
from typing import Optional

import gradio as gr
from google import genai
from google.genai import types


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_CLIENT: Optional[genai.Client] = None


SYSTEM_PROMPT = """
You are a multimodal sentiment analysis assistant.

Analyze the given text and/or video/audio content and classify the overall
sentiment into exactly one label:

- negative
- neutral
- positive

Return this exact format:

Sentiment: <negative|neutral|positive>
Confidence: <low|medium|high>
Reason: <one or two short sentences>
Evidence: <brief mention of text, voice, or visual cues>

If the input is unclear, choose neutral with low confidence.
"""


def get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get one from "
                "https://aistudio.google.com/app/apikey"
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def mime_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def file_part(path: str) -> types.Part:
    return types.Part.from_bytes(
        data=Path(path).read_bytes(),
        mime_type=mime_type(path),
    )


def analyze_with_gemini(text: str = "", video_path: Optional[str] = None) -> str:
    clean_text = (text or "").strip()

    if not clean_text and not video_path:
        return "Please enter text or upload/record a video first."

    parts = [SYSTEM_PROMPT.strip()]

    if clean_text:
        parts.append(f"\nUser text/transcript:\n{clean_text}")

    if video_path:
        parts.append("\nUploaded or recorded video:")
        parts.append(file_part(video_path))

    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300,
        ),
    )

    return response.text or "No response returned from Gemini."


def result_html(result: str) -> str:
    safe = html.escape(result or "").replace("\n", "<br>")
    return f"""
    <div class="result-card">
      <h3>Analysis Result</h3>
      <p>{safe}</p>
    </div>
    """


def analyze(text: str, video_file: Optional[str], webcam_video: Optional[str]) -> str:
    selected_video = webcam_video or video_file
    result = analyze_with_gemini(text=text, video_path=selected_video)
    return result_html(result)


def clear_video_when_text_changes(text: str):
    if (text or "").strip():
        return gr.update(value=None), gr.update(value=None)
    return gr.update(), gr.update()


def clear_text_when_video_added(video_file, webcam_video):
    if video_file is not None or webcam_video is not None:
        return gr.update(value="")
    return gr.update()


CSS = """
body {
  background: #f7f7fb;
}

.gradio-container {
  max-width: 980px !important;
  margin: auto !important;
}

.hero {
  padding: 18px 22px;
  border-radius: 18px;
  background: linear-gradient(135deg, #ffffff, #eef2ff);
  border: 1px solid #e5e7eb;
  margin-bottom: 18px;
}

.hero h1 {
  margin: 0 0 6px 0;
  font-size: 30px;
}

.hero p {
  margin: 0;
  color: #4b5563;
}

.result-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 18px;
  min-height: 150px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.result-card h3 {
  margin-top: 0;
}
"""


with gr.Blocks(css=CSS, title="Multimodal Sentiment Analysis - Gemini") as demo:
    gr.HTML(
        """
        <div class="hero">
          <h1>Multimodal Sentiment Analysis</h1>
          <p>Analyze sentiment from text, uploaded video, or recorded video using Gemini.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Type a sentence here, or leave empty and use video.",
                lines=5,
            )

            video_upload = gr.Video(
                label="Upload recorded video",
                sources=["upload"],
            )

            webcam_input = gr.Video(
                label="Record video",
                sources=["webcam"],
            )

            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.ClearButton(
                    components=[text_input, video_upload, webcam_input],
                    value="Clear",
                )

        with gr.Column(scale=1):
            output = gr.HTML(
                """
                <div class="result-card">
                  <h3>Analysis Result</h3>
                  <p>Your result will appear here.</p>
                </div>
                """
            )

    text_input.change(
        clear_video_when_text_changes,
        inputs=[text_input],
        outputs=[video_upload, webcam_input],
    )

    video_upload.change(
        clear_text_when_video_added,
        inputs=[video_upload, webcam_input],
        outputs=[text_input],
    )

    webcam_input.change(
        clear_text_when_video_added,
        inputs=[video_upload, webcam_input],
        outputs=[text_input],
    )

    analyze_btn.click(
        analyze,
        inputs=[text_input, video_upload, webcam_input],
        outputs=[output],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
