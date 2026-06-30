"""Simple Gradio interface for multimodal sentiment analysis."""

import html
import os
import sys
import time
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.predictor import SentimentPredictor

try:
    from inference.gemini_predictor import GeminiFallbackPredictor
except Exception as exc:
    GeminiFallbackPredictor = None
    print(f'Gemini fallback import unavailable: {exc}')

MODEL_PATH = os.environ.get('MODEL_PATH', r'D:\data\multimodal\results\best_model.pt')
CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config.json')
USE_GEMINI_FALLBACK = os.environ.get('USE_GEMINI_FALLBACK', 'true').lower() == 'true'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_CONFIDENCE_THRESHOLD = float(os.environ.get('GEMINI_CONFIDENCE_THRESHOLD', '0.55')) * 100.0

print('Loading SentimentPredictor...')
predictor = SentimentPredictor(model_path=MODEL_PATH, config_path=CONFIG_PATH)
print('Predictor ready')

gemini_predictor = None
if USE_GEMINI_FALLBACK and GEMINI_API_KEY and GeminiFallbackPredictor is not None:
    try:
        gemini_predictor = GeminiFallbackPredictor(
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL,
        )
        print(f'Gemini fallback ready: {GEMINI_MODEL}')
    except Exception as exc:
        print(f'Gemini fallback disabled: {exc}')
elif USE_GEMINI_FALLBACK:
    print('Gemini fallback disabled: GEMINI_API_KEY not set')


CSS = """
.gradio-container {
  max-width: 1080px !important;
  margin: 0 auto !important;
  background: #f7f8fc !important;
}
.header {
  text-align: center;
  padding: 26px 12px 20px;
}
.header h1 {
  color: #172033;
  font-size: 2rem;
  margin: 0 0 8px;
}
.header p {
  color: #667085;
  margin: 0;
}
.card {
  background: white !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 16px !important;
  padding: 16px !important;
  box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06);
}
.primary-btn, .secondary-btn {
  border-radius: 10px !important;
  min-height: 44px !important;
}
.result-card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06);
}
.result-row {
  display: flex;
  align-items: center;
  gap: 16px;
}
.result-emoji { font-size: 3rem; }
.result-label { font-size: 1.8rem; font-weight: 750; }
.result-meta { color: #667085; margin-top: 4px; }
.confidence-track {
  height: 9px;
  margin-top: 20px;
  background: #eef0f4;
  border-radius: 999px;
  overflow: hidden;
}
.confidence-fill { height: 100%; border-radius: 999px; }
.transcript-card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 14px;
  color: #344054;
  min-height: 100px;
  box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06);
}
.transcript-card h3 { color: #172033; margin: 0 0 10px; }
.placeholder { color: #98a2b3; }
.error-card {
  color: #b42318;
  background: #fff4f2;
  border: 1px solid #fecdca;
  border-radius: 12px;
  padding: 16px;
}
"""


def fallback_note_html(result: dict) -> str:
    details = []
    if result.get('reason'):
        details.append(html.escape(str(result['reason'])))
    if result.get('evidence'):
        details.append(html.escape(str(result['evidence'])))
    if not details:
        return ''
    return f'<div class="result-meta" style="margin-top:12px">{" ".join(details)}</div>'


def result_html(result: dict) -> str:
    color = result['color']
    confidence = result['confidence']
    source = html.escape(result.get('provider', 'custom_model').replace('_', ' ').title())
    return f"""
    <div class="result-card">
      <div class="result-row">
        <div class="result-emoji">{result['emoji']}</div>
        <div>
          <div class="result-label" style="color:{color}">{result['label']}</div>
          <div class="result-meta">Confidence: {confidence:.1f}%</div>
          <div class="result-meta">Source: {source}</div>
        </div>
      </div>
      {fallback_note_html(result)}
      <div class="confidence-track">
        <div class="confidence-fill" style="width:{confidence}%;background:{color}"></div>
      </div>
    </div>
    """


def transcript_html(text: str) -> str:
    safe_text = html.escape((text or '').strip())
    if not safe_text:
        safe_text = '<span class="placeholder">No speech transcript was detected.</span>'
    return f'<div class="transcript-card"><h3>Transcript</h3><div>{safe_text}</div></div>'


def toggle_text_input(media_path):
    if media_path:
        return gr.update(
            value='',
            interactive=False,
            placeholder='Text input is disabled while a video is selected.',
        )
    return gr.update(
        interactive=True,
        placeholder='Type text here when no video is selected.',
    )


def should_use_gemini(result: dict) -> bool:
    if gemini_predictor is None:
        return False
    if 'error' in result:
        return True
    return float(result.get('confidence', 100.0)) < GEMINI_CONFIDENCE_THRESHOLD


def run_gemini_fallback(text: str = '', media_path: str | None = None) -> dict | None:
    if gemini_predictor is None:
        return None
    fallback = gemini_predictor.predict(text=text, video_path=media_path)
    if 'error' in fallback:
        print(f"Gemini fallback failed: {fallback.get('message', 'unknown error')}", flush=True)
        return None
    return fallback


def analyze(media_path, text, progress=gr.Progress()):
    started = time.perf_counter()
    text = (text or '').strip()

    if media_path:
        progress(0.1, desc='Analyzing video')
        result = predictor.predict_from_video(str(media_path), mode='upload')
        result.setdefault('provider', 'custom_model')
        if should_use_gemini(result):
            fallback = run_gemini_fallback(
                text=result.get('transcript', ''),
                media_path=str(media_path),
            )
            if fallback is not None:
                result = fallback
    elif text:
        progress(0.2, desc='Analyzing text')
        result = predictor.predict_from_text(text)
        result.setdefault('provider', 'custom_model')
        if should_use_gemini(result):
            fallback = run_gemini_fallback(text=text, media_path=None)
            if fallback is not None:
                result = fallback
    else:
        error = '<div class="error-card">Record or upload a video, or enter some text.</div>'
        return '', error

    if 'error' in result:
        message = html.escape(str(result.get('message', 'Inference failed.')))
        return '', f'<div class="error-card">{message}</div>'

    print(
        f'[TIMING][gradio] unified_analysis_total={time.perf_counter() - started:.3f}s',
        flush=True,
    )
    progress(1.0, desc='Complete')
    return transcript_html(result.get('transcript', text)), result_html(result)


def clear_all():
    return (
        None,
        gr.update(value='', interactive=True, placeholder='Type text here when no video is selected.'),
        '<div class="transcript-card"><h3>Transcript</h3><div class="placeholder">Transcript will appear here.</div></div>',
        '<div class="result-card"><div class="placeholder">Prediction will appear here.</div></div>',
    )


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue='indigo', neutral_hue='slate'),
    css=CSS,
    title='Multimodal Sentiment Analysis',
) as demo:
    gr.HTML("""
    <div class="header">
      <h1>Multimodal Sentiment Analysis</h1>
      <p>Upload or record a video, or analyze text.</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, elem_classes='card'):
            media_input = gr.Video(
                label='Video',
                sources=['upload', 'webcam'],
                format=None,
                include_audio=True,
                webcam_options=gr.WebcamOptions(
                    mirror=False,
                    constraints={
                        'video': {
                            'width': {'ideal': 640},
                            'height': {'ideal': 480},
                            'frameRate': {'ideal': 24, 'max': 30},
                        },
                        'audio': True,
                    },
                ),
                height=340,
            )
            text_input = gr.Textbox(
                label='Text',
                placeholder='Type text here when no video is selected.',
                lines=3,
            )
            with gr.Row():
                analyze_button = gr.Button(
                    'Analyze', variant='primary', elem_classes='primary-btn'
                )
                clear_button = gr.Button(
                    'Clear', variant='secondary', elem_classes='secondary-btn'
                )

        with gr.Column(scale=5):
            transcript_output = gr.HTML(
                '<div class="transcript-card"><h3>Transcript</h3>'
                '<div class="placeholder">Transcript will appear here.</div></div>'
            )
            result_output = gr.HTML(
                '<div class="result-card"><div class="placeholder">'
                'Prediction will appear here.</div></div>'
            )

    media_input.change(
        fn=toggle_text_input,
        inputs=[media_input],
        outputs=[text_input],
        queue=False,
    )
    analyze_button.click(
        fn=analyze,
        inputs=[media_input, text_input],
        outputs=[transcript_output, result_output],
        concurrency_limit=1,
        show_progress='full',
    )
    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[media_input, text_input, transcript_output, result_output],
        queue=False,
    )

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False,
    )