"""Unified Gradio interface for multimodal sentiment analysis."""

import os
import sys
import time
import html
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import SentimentPredictor

MODEL_PATH = os.environ.get('MODEL_PATH', r'D:\data\multimodal\results\best_model.pt')
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config.json')

print('Loading SentimentPredictor...')
predictor = SentimentPredictor(model_path=MODEL_PATH, config_path=CONFIG_PATH)
print('✅ Predictor ready')


CSS = """
:root {
  --panel: rgba(17, 24, 39, 0.76);
  --line: rgba(148, 163, 184, 0.18);
  --muted: #94a3b8;
}
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  background:
    radial-gradient(circle at 15% 15%, rgba(59,130,246,.16), transparent 32%),
    radial-gradient(circle at 85% 5%, rgba(168,85,247,.14), transparent 30%),
    #070b14 !important;
}
.hero {
  padding: 34px 32px;
  border: 1px solid var(--line);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(30,41,59,.94), rgba(15,23,42,.82));
  box-shadow: 0 24px 70px rgba(0,0,0,.28);
  margin-bottom: 18px;
}
.hero h1 { margin: 0; color: #f8fafc; font-size: 2.35rem; letter-spacing: -.04em; }
.hero p { color: #cbd5e1; max-width: 760px; font-size: 1.02rem; margin: 12px 0 0; }
.eyebrow { color: #60a5fa; font-weight: 700; text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; }
.panel { border: 1px solid var(--line) !important; border-radius: 20px !important; background: var(--panel) !important; padding: 8px !important; }
.primary-btn { border-radius: 14px !important; min-height: 48px !important; font-weight: 700 !important; }
.secondary-btn { border-radius: 14px !important; min-height: 48px !important; }
.result-card {
  padding: 24px;
  border: 1px solid var(--line);
  border-radius: 20px;
  background: linear-gradient(145deg, rgba(15,23,42,.95), rgba(30,41,59,.82));
  min-height: 190px;
}
.score-row { display: flex; align-items: center; gap: 16px; }
.score-emoji { font-size: 3.5rem; }
.score-label { font-size: 2rem; font-weight: 800; letter-spacing: -.03em; }
.score-meta { color: var(--muted); margin-top: 4px; }
.confidence-track { height: 10px; border-radius: 99px; background: rgba(148,163,184,.16); margin-top: 22px; overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 99px; }
.info-card { padding: 16px 18px; border: 1px solid var(--line); border-radius: 16px; background: rgba(15,23,42,.72); color: #cbd5e1; }
.info-card strong { color: #f8fafc; }
.latency { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .84rem; color: #93c5fd; }
.footer-note { text-align: center; color: #64748b; font-size: .82rem; padding: 18px; }
"""


def result_html(result: dict) -> str:
    color = result['color']
    confidence = result['confidence']
    return f"""
    <div class="result-card">
      <div class="score-row">
        <div class="score-emoji">{result['emoji']}</div>
        <div>
          <div class="score-label" style="color:{color}">{result['label']}</div>
          <div class="score-meta">Confidence {confidence:.1f}% · score {result['score']:+.0f}</div>
        </div>
      </div>
      <div class="confidence-track">
        <div class="confidence-fill" style="width:{confidence}%;background:{color}"></div>
      </div>
    </div>
    """


def modality_html(result: dict) -> str:
    def state(active, name):
        return f"{'●' if active else '○'} {name}"

    warnings = result.get('warnings', [])
    warning_html = ''.join(f'<div>⚠ {html.escape(str(warning))}</div>' for warning in warnings)
    return f"""
    <div class="info-card">
      <strong>Signals used</strong><br><br>
      {state(result.get('has_text', False), 'Text / speech')} &nbsp;&nbsp;
      {state(result.get('has_audio', False), 'Audio')} &nbsp;&nbsp;
      {state(result.get('has_vision', False), 'Vision')}
      <div style="color:#fbbf24;margin-top:10px">{warning_html}</div>
    </div>
    """


def transcript_html(text: str) -> str:
    safe_text = html.escape((text or '').strip())
    if not safe_text:
        safe_text = 'No speech transcript was detected.'
    return f'<div class="info-card"><strong>Transcript</strong><p style="margin-bottom:0">{safe_text}</p></div>'


def latency_html(result: dict, total_seconds: float) -> str:
    timings = result.get('timings', {})
    parts = []
    for key in ('audio_features', 'vision_features', 'normalization', 'feature_extraction_total'):
        if key in timings:
            parts.append(f'{key.replace("_", " ")}: {timings[key]:.3f}s')
    parts.append(f'end to end: {total_seconds:.3f}s')
    return '<div class="info-card latency"><strong>Performance</strong><br>' + '<br>'.join(parts) + '</div>'


def analyze(media_path, text, progress=gr.Progress()):
    started = time.perf_counter()
    text = (text or '').strip()

    if media_path:
        progress(0.08, desc='Reading video and audio')
        result = predictor.predict_from_video(str(media_path), mode='upload')
        source = 'Recorded or uploaded media'
    elif text:
        progress(0.2, desc='Analyzing text')
        result = predictor.predict_from_text(text)
        source = 'Text only'
    else:
        message = '<div class="info-card" style="color:#f87171">Add text or record/upload a video first.</div>'
        return message, '', '', '', 'Waiting for input'

    if 'error' in result:
        message = html.escape(str(result.get('message', 'Inference failed')))
        error = f'<div class="info-card" style="color:#f87171"><strong>Unable to analyze</strong><br>{message}</div>'
        return error, '', '', '', 'Analysis failed'

    total_seconds = time.perf_counter() - started
    print(f'[TIMING][gradio] unified_analysis_total={total_seconds:.3f}s', flush=True)
    progress(1.0, desc='Complete')

    return (
        result_html(result),
        transcript_html(result.get('transcript', text)),
        modality_html(result),
        latency_html(result, total_seconds),
        f'{source} · completed in {total_seconds:.2f}s',
    )


def clear_all():
    return None, '', '', '', '', '', 'Ready'


HEADER = """
<div class="hero">
  <div class="eyebrow">Multimodal AI · MELD</div>
  <h1>Read the feeling behind the moment.</h1>
  <p>Record a short clip, upload an existing video, or enter text. The model combines language, vocal cues, and visual context into one sentiment prediction.</p>
</div>
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue='blue', neutral_hue='slate'), css=CSS, title='Multimodal Sentiment') as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, elem_classes='panel'):
            gr.Markdown('### Add your input')
            media_input = gr.Video(
                label='Upload or record a video',
                sources=['upload', 'webcam'],
                format=None,
                include_audio=True,
                webcam_options=gr.WebcamOptions(
                    mirror=False,
                    constraints={
                        'video': {'width': 640, 'height': 480, 'frameRate': {'ideal': 24, 'max': 30}},
                        'audio': True,
                    },
                ),
                height=360,
            )
            text_input = gr.Textbox(
                label='Or analyze text only',
                placeholder='Type a sentence when no video is provided…',
                lines=3,
            )
            with gr.Row():
                analyze_button = gr.Button('Analyze sentiment', variant='primary', elem_classes='primary-btn')
                clear_button = gr.Button('Clear', variant='secondary', elem_classes='secondary-btn')
            status = gr.Textbox(value='Ready', label='Status', interactive=False)

        with gr.Column(scale=5):
            gr.Markdown('### Result')
            result_output = gr.HTML('<div class="result-card"><div class="score-meta">Your prediction will appear here.</div></div>')
            modality_output = gr.HTML()

    with gr.Row():
        with gr.Column(scale=7):
            transcript_output = gr.HTML()
        with gr.Column(scale=4):
            latency_output = gr.HTML()

    gr.HTML('<div class="footer-note">RoBERTa-base · Wav2Vec2 · CLIP ViT-B/32 · Cross-modal Transformer</div>')

    analyze_button.click(
        fn=analyze,
        inputs=[media_input, text_input],
        outputs=[result_output, transcript_output, modality_output, latency_output, status],
        concurrency_limit=1,
        show_progress='full',
    )
    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[media_input, text_input, result_output, transcript_output, modality_output, latency_output, status],
        queue=False,
    )

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False,
    )