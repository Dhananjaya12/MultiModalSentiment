"""
app/app.py

Gradio UI for Multimodal Sentiment Analysis.
Three tabs: Text, Video Upload, Live Webcam.

Deploy on Hugging Face Spaces:
  - Set MODEL_PATH env var to your model file path
  - Set CONFIG_PATH env var to your config.json path
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import SentimentPredictor

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get('MODEL_PATH',  r'D:\data\multimodal\results\best_model.pt')
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config.json')

# ── Load predictor once at startup ───────────────────────────────
print('Loading SentimentPredictor...')
predictor = SentimentPredictor(
    model_path  = MODEL_PATH,
    config_path = CONFIG_PATH,
)
print('✅ Predictor ready')


# ── Helper: build sentiment gauge HTML ───────────────────────────

def build_gauge_html(score: float, label: str, confidence: float, color: str, emoji: str) -> str:
    """Build a visual sentiment gauge as HTML."""
    pct = int((score + 1) / 2 * 100)  # map -1..1 → 0..100%
    return f"""
    <div style="font-family: 'Segoe UI', sans-serif; padding: 16px; 
                background: #1a1a2e; border-radius: 12px; color: white;">
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
        <span style="font-size:2.5rem;">{emoji}</span>
        <div>
          <div style="font-size:1.8rem; font-weight:700; color:{color};">{label}</div>
          <div style="font-size:0.9rem; color:#aaa;">Confidence: {confidence:.1f}%</div>
        </div>
      </div>
      <div style="background:#333; border-radius:8px; height:20px; overflow:hidden;">
        <div style="width:{pct}%; background:{color}; height:100%; 
                    border-radius:8px; transition:width 0.5s ease;">
        </div>
      </div>
      <div style="display:flex; justify-content:space-between; 
                  font-size:0.75rem; color:#888; margin-top:4px;">
        <span>Negative</span><span>Neutral</span><span>Positive</span>
      </div>
      <div style="text-align:center; font-size:0.85rem; color:#aaa; margin-top:8px;">
        Score: {score:+.3f}
      </div>
    </div>
    """


def build_modality_html(has_audio: bool, has_vision: bool, has_text: bool, warnings: list) -> str:
    """Build modality availability indicator."""
    def icon(active): return '✅' if active else '❌'
    warning_html = ''
    if warnings:
        items = ''.join(f'<li>{w}</li>' for w in warnings)
        warning_html = f'<ul style="color:#FFBB33;margin:8px 0;padding-left:20px;">{items}</ul>'

    return f"""
    <div style="font-family: monospace; padding:12px; background:#0d0d1a; 
                border-radius:8px; color:white; margin-top:8px;">
      <b>Modality Detection</b>
      <div style="margin-top:8px;">
        {icon(has_text)}  Text/Speech &nbsp;&nbsp;
        {icon(has_audio)} Audio &nbsp;&nbsp;
        {icon(has_vision)} Vision (Face)
      </div>
      {warning_html}
    </div>
    """


def build_transcript_html(utterances: list) -> str:
    """Build color-coded transcript from utterance list."""
    if not utterances:
        return '<p style="color:#888;">No transcript available.</p>'

    lines = []
    for u in utterances:
        color = u.get('color', '#FFBB33')
        text  = u.get('text', '')
        label = u.get('label', '')
        start = u.get('start', 0)
        emoji = u.get('emoji', '')
        lines.append(
            f'<div style="padding:6px 10px; margin:4px 0; border-left:4px solid {color}; '
            f'background:rgba(255,255,255,0.05); border-radius:4px;">'
            f'<span style="color:#888; font-size:0.8rem;">[{start:.1f}s]</span> '
            f'<span style="color:white;">{text}</span> '
            f'<span style="color:{color}; font-size:0.8rem;">{emoji} {label}</span>'
            f'</div>'
        )
    return '<div style="font-family: sans-serif;">' + ''.join(lines) + '</div>'


# ── Tab 1: Text Analysis ──────────────────────────────────────────

def analyze_text(text: str):
    request_started = time.perf_counter()
    if not text or not text.strip():
        return (
            '<p style="color:#FF4444;">Please enter some text.</p>',
            None, None
        )

    result = predictor.predict_from_text(text)
    print(f'[TIMING][gradio] analyze_text_total={time.perf_counter() - request_started:.3f}s', flush=True)

    if 'error' in result:
        return (
            f'<p style="color:#FF4444;">Error: {result["message"]}</p>',
            None, None
        )

    gauge = build_gauge_html(
        score      = result['score'],
        label      = result['label'],
        confidence = result['confidence'],
        color      = result['color'],
        emoji      = result['emoji'],
    )
    modality = build_modality_html(
        has_audio  = result['has_audio'],
        has_vision = result['has_vision'],
        has_text   = result['has_text'],
        warnings   = result['warnings'],
    )
    note = '📝 Analysis based on text only. Upload a video for full multimodal analysis.'

    return gauge, modality, note


# ── Tab 2: Video Upload ───────────────────────────────────────────

def analyze_video(file):
    request_started = time.perf_counter()
    if file is None:
        return (
            '<p style="color:#FF4444;">Please upload a video file.</p>',
            None, None, None, None
        )

    video_path = file.name if hasattr(file, 'name') else str(file)

    # Overall prediction
    result = predictor.predict_from_video(video_path, mode='upload')
    overall_seconds = time.perf_counter() - request_started

    if 'error' in result:
        msg = result.get('message', 'Unknown error')
        return (
            f'<p style="color:#FF4444;">Error: {msg}</p>',
            None, None, None, None
        )

    gauge = build_gauge_html(
        score      = result['score'],
        label      = result['label'],
        confidence = result['confidence'],
        color      = result['color'],
        emoji      = result['emoji'],
    )

    # Utterance timeline
    timeline_started = time.perf_counter()
    utterances = predictor.predict_utterances(video_path, mode='upload')
    timeline_seconds = time.perf_counter() - timeline_started
    print(
        f'[TIMING][gradio] video_overall={overall_seconds:.3f}s | '
        f'video_timeline={timeline_seconds:.3f}s | '
        f'analyze_video_total={time.perf_counter() - request_started:.3f}s',
        flush=True,
    )

    # Build timeline dataframe data
    if utterances:
        import pandas as pd
        df = pd.DataFrame([{
            'Time (s)':   f"{u['start']:.1f} – {u['end']:.1f}",
            'Text':       u['text'],
            'Sentiment':  f"{u['emoji']} {u['label']}",
            'Score':      f"{u['score']:+.2f}",
            'Confidence': f"{u['confidence']:.1f}%",
        } for u in utterances])
    else:
        import pandas as pd
        df = pd.DataFrame(columns=['Time (s)', 'Text', 'Sentiment', 'Score', 'Confidence'])

    transcript_html = build_transcript_html(utterances)
    modality_html   = build_modality_html(
        has_audio  = result['has_audio'],
        has_vision = result['has_vision'],
        has_text   = result['has_text'],
        warnings   = result['warnings'],
    )

    return gauge, df, transcript_html, modality_html, None


# ── Tab 3: Live Webcam ────────────────────────────────────────────

webcam_history = []

def analyze_webcam_clip(video):
    """
    Called when user records a clip from webcam.
    Uses whisper-tiny for fast inference.
    """
    global webcam_history
    request_started = time.perf_counter()

    if video is None:
        return (
            '<p style="color:#888;">Record a clip to analyze.</p>',
            None
        )

    result = predictor.predict_from_video(video, mode='webcam')
    print(f'[TIMING][gradio] analyze_webcam_total={time.perf_counter() - request_started:.3f}s', flush=True)

    if 'error' in result:
        return (
            f'<p style="color:#FF4444;">Error: {result.get("message", "")}</p>',
            None
        )

    # Add to rolling history (last 10)
    webcam_history.append({
        'text':   result.get('transcript', '(no speech)'),
        'label':  result['label'],
        'score':  result['score'],
        'color':  result['color'],
        'emoji':  result['emoji'],
        'confidence': result['confidence'],
    })
    webcam_history = webcam_history[-10:]

    gauge = build_gauge_html(
        score      = result['score'],
        label      = result['label'],
        confidence = result['confidence'],
        color      = result['color'],
        emoji      = result['emoji'],
    )

    # Rolling history table
    import pandas as pd
    df = pd.DataFrame([{
        'Utterance':  h['text'][:60] + ('...' if len(h['text']) > 60 else ''),
        'Sentiment':  f"{h['emoji']} {h['label']}",
        'Score':      f"{h['score']:+.2f}",
        'Confidence': f"{h['confidence']:.1f}%",
    } for h in reversed(webcam_history)])

    return gauge, df


def clear_webcam_history():
    global webcam_history
    webcam_history = []
    return None, None


# ── Build Gradio App ──────────────────────────────────────────────

HEADER = """
<div style="text-align:center; padding:20px; 
            background:linear-gradient(135deg,#1a1a2e,#16213e);
            border-radius:12px; margin-bottom:16px;">
  <h1 style="color:white; font-size:2rem; margin:0;">
    🎭 Multimodal Sentiment Analyzer
  </h1>
  <p style="color:#aaa; margin:8px 0 0;">
    Analyzes sentiment from <b>text</b>, <b>audio</b>, and <b>facial expressions</b> 
    using a cross-modal transformer trained on MELD dataset.
  </p>
</div>
"""

TEXT_EXAMPLES = [
    ["I absolutely love this product, it works perfectly!"],
    ["This is the worst experience I've ever had."],
    ["The meeting was okay, nothing special happened."],
    ["I can't believe how amazing this turned out!"],
    ["I'm not sure how I feel about this decision."],
]

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue='blue'),
    title='Multimodal Sentiment Analyzer',
) as demo:

    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Text ───────────────────────────────────────────
        with gr.TabItem('📝 Text Analysis'):
            gr.Markdown(
                '### Analyze sentiment from text\n'
                'Type or paste any text. Results are instant.'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        lines       = 4,
                        placeholder = 'Type or paste text here...',
                        label       = 'Input Text',
                    )
                    text_btn = gr.Button('🔍 Analyze', variant='primary')
                    gr.Examples(
                        examples   = TEXT_EXAMPLES,
                        inputs     = [text_input],
                        label      = 'Example Sentences',
                    )
                with gr.Column(scale=2):
                    text_gauge    = gr.HTML(label='Sentiment')
                    text_modality = gr.HTML(label='Modalities')
                    text_note     = gr.Textbox(label='Note', interactive=False)

            text_btn.click(
                fn      = analyze_text,
                inputs  = [text_input],
                outputs = [text_gauge, text_modality, text_note],
            )

        # ── Tab 2: Video Upload ───────────────────────────────────
        with gr.TabItem('🎬 Video Upload'):
            gr.Markdown(
                '### Analyze sentiment from a video or audio file\n'
                'Upload an `.mp4`, `.mov`, `.avi`, `.wav`, or `.mp3` file. '
                'Analysis includes per-utterance timeline.'
            )
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.File(
                        label      = 'Upload Video / Audio',
                        file_types = ['.mp4', '.mov', '.avi', '.wav', '.mp3'],
                    )
                    video_btn = gr.Button('🔍 Analyze', variant='primary')
                with gr.Column(scale=2):
                    video_gauge    = gr.HTML(label='Overall Sentiment')
                    video_modality = gr.HTML(label='Modalities Detected')

            gr.Markdown('### Utterance Timeline')
            video_df = gr.Dataframe(
                headers    = ['Time (s)', 'Text', 'Sentiment', 'Score', 'Confidence'],
                label      = 'Per-Utterance Results',
                interactive= False,
            )
            gr.Markdown('### Color-Coded Transcript')
            video_transcript = gr.HTML(label='Transcript')
            video_warn       = gr.HTML()   # hidden warnings

            video_btn.click(
                fn      = analyze_video,
                inputs  = [video_input],
                outputs = [video_gauge, video_df, video_transcript, video_modality, video_warn],
            )

        # ── Tab 3: Webcam ─────────────────────────────────────────
        with gr.TabItem('📹 Live Webcam'):
            gr.Markdown(
                '### Analyze sentiment from live webcam\n'
                'Record a short clip (3-10 seconds). '
                'Uses faster Whisper-tiny model for speed.\n\n'
                '> **Tip:** Speak clearly for 3-5 seconds, then stop recording.'
            )
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Video(
                        sources = ['webcam'],
                        label   = 'Record Clip',
                    )
                    with gr.Row():
                        webcam_btn   = gr.Button('🔍 Analyze Clip', variant='primary')
                        webcam_clear = gr.Button('🗑️ Clear History', variant='secondary')
                with gr.Column(scale=2):
                    webcam_gauge = gr.HTML(label='Current Sentiment')

            gr.Markdown('### Last 10 Utterances')
            webcam_df = gr.Dataframe(
                headers    = ['Utterance', 'Sentiment', 'Score', 'Confidence'],
                label      = 'Rolling History',
                interactive= False,
            )

            webcam_btn.click(
                fn      = analyze_webcam_clip,
                inputs  = [webcam_input],
                outputs = [webcam_gauge, webcam_df],
            )
            webcam_clear.click(
                fn      = clear_webcam_history,
                inputs  = [],
                outputs = [webcam_gauge, webcam_df],
            )

    # ── Footer ────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding:16px; color:#888; font-size:0.85rem;">
      Trained on <b>MELD dataset</b> (Friends TV series) · 
      RoBERTa-large + MediaPipe + librosa · 
      3-class sentiment: Negative / Neutral / Positive
    </div>
    """)


if __name__ == '__main__':
    demo.launch(
        server_name = '0.0.0.0',
        server_port = 7860,
        share       = False,
    )