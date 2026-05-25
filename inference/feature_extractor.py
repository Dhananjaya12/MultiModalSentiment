"""
inference/feature_extractor.py

Feature extraction for inference — MUST match data prep notebook exactly.

Settings that must never change (v2):
  sr          = 16000
  audio_dim   = 768  (wav2vec2-base hidden size)
  vision_dim  = 512  (CLIP ViT-B/32 output)
  seq_len     = 300
  frame_step  = 3    (every 3rd frame for CLIP)
  wav2vec2    = facebook/wav2vec2-base-960h
  clip        = ViT-B-32, openai weights
"""

import os
import cv2
import numpy as np
import torch
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Constants — must match data prep notebook ─────────────────────
SEQ_LEN    = 300
AUDIO_DIM  = 768   # wav2vec2-base hidden size
VISION_DIM = 512   # CLIP ViT-B/32
SR         = 16000
FRAME_STEP = 3     # same as data prep — every 3rd frame

WAV2VEC_MODEL = 'facebook/wav2vec2-base-960h'
CLIP_MODEL    = 'ViT-B-32'
CLIP_WEIGHTS  = 'openai'

# ── Lazy-loaded model globals ─────────────────────────────────────
_wav2vec_processor = None
_wav2vec_model     = None
_clip_model        = None
_clip_preprocess   = None
_device            = None


def _get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


def _get_wav2vec():
    """Lazy load wav2vec2 — only when first audio extraction is needed."""
    global _wav2vec_processor, _wav2vec_model
    if _wav2vec_processor is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        print(f'Loading {WAV2VEC_MODEL}...')
        _wav2vec_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
        _wav2vec_model     = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL).to(_get_device())
        _wav2vec_model.eval()
        print('✅ wav2vec2 ready')
    return _wav2vec_processor, _wav2vec_model


def _get_clip():
    """Lazy load CLIP — only when first vision extraction is needed."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import open_clip
        print(f'Loading CLIP {CLIP_MODEL}...')
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_WEIGHTS
        )
        _clip_model = _clip_model.to(_get_device())
        _clip_model.eval()
        print('✅ CLIP ready')
    return _clip_model, _clip_preprocess


# ── Audio feature extraction ──────────────────────────────────────

def extract_audio_features(video_path, seq_len=SEQ_LEN, audio_dim=AUDIO_DIM):
    import tempfile, librosa
    fname      = os.path.basename(str(video_path)) + '_infer.wav'
    audio_path = os.path.join(tempfile.gettempdir(), fname)

    try:
        os.system(
            f'ffmpeg -i "{video_path}" -ac 1 -ar {SR} '
            f'"{audio_path}" -y -loglevel quiet'
        )
        if not os.path.exists(audio_path):
            return np.zeros((seq_len, audio_dim), dtype=np.float32)

        y, sr = librosa.load(audio_path, sr=SR)
        if len(y) < 400:
            y = np.pad(y, (0, 400 - len(y)))

        processor, model = _get_wav2vec()
        device = _get_device()
        inputs = processor(y, sampling_rate=SR, return_tensors='pt', padding=True)

        with torch.no_grad():
            out = model(inputs.input_values.to(device))

        features = out.last_hidden_state.squeeze(0).cpu().numpy()

        T = features.shape[0]
        if T < seq_len:
            features = np.vstack([features, np.zeros((seq_len-T, audio_dim), dtype=np.float32)])
        else:
            features = features[:seq_len]

        return features.astype(np.float32)

    except Exception as e:
        return np.zeros((seq_len, audio_dim), dtype=np.float32)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def extract_vision_features(video_path, seq_len=SEQ_LEN, vision_dim=VISION_DIM):
    from PIL import Image
    try:
        clip_m, clip_prep = _get_clip()
        device = _get_device()
        cap, frames, fc = cv2.VideoCapture(str(video_path)), [], 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fc % FRAME_STEP == 0:
                try:
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil   = Image.fromarray(rgb)
                    img_t = clip_prep(pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        feat = clip_m.encode_image(img_t)
                        feat = feat.squeeze(0).cpu().numpy().astype(np.float32)

                    # Force exactly vision_dim — matches notebook exactly
                    feat = feat.flatten()
                    if len(feat) >= vision_dim:
                        feat = feat[:vision_dim]
                    else:
                        feat = np.pad(feat, (0, vision_dim - len(feat)))

                    frames.append(feat)

                except Exception:
                    # Bad frame — zeros, keep consistent shape
                    frames.append(np.zeros(vision_dim, dtype=np.float32))

            fc += 1

        cap.release()

        if len(frames) == 0:
            return np.zeros((seq_len, vision_dim), dtype=np.float32)

        features = np.stack(frames)
        T = features.shape[0]
        if T < seq_len:
            features = np.vstack([features, np.zeros((seq_len-T, vision_dim), dtype=np.float32)])
        else:
            features = features[:seq_len]

        return features.astype(np.float32)

    except Exception as e:
        return np.zeros((seq_len, vision_dim), dtype=np.float32)


# ── Text feature extraction ───────────────────────────────────────

def extract_text_features(
    text:       str,
    tokenizer,
    max_len:    int = 128,
):
    """
    Tokenize text with RoBERTa tokenizer.

    Returns:
        input_ids:      torch.Tensor shape (max_len,) dtype=long
        attention_mask: torch.Tensor shape (max_len,) dtype=long

    Edge case:
        Empty string → zero input_ids, ones attention_mask
        (ones mask prevents transformer from crashing)
    """
    text = text.strip() if text else ''

    if len(text) == 0:
        input_ids      = torch.zeros(max_len, dtype=torch.long)
        attention_mask = torch.ones(max_len,  dtype=torch.long)
        return input_ids, attention_mask

    enc = tokenizer(
        text,
        max_length     = max_len,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt',
    )
    return (
        enc['input_ids'].squeeze(0),
        enc['attention_mask'].squeeze(0),
    )


# ── Normalization — must match dataloader.py exactly ─────────────

def apply_normalization(
    audio:  np.ndarray,
    vision: np.ndarray,
) -> tuple:
    """
    Per-sample normalization.
    MUST match dataloader.py:
        audio  = (audio  - audio.mean())  / (audio.std()  + 1e-8)
        vision = (vision - vision.mean()) / (vision.std() + 1e-8)

    Skips normalization if std is near zero (all-zeros input).
    """
    if audio.std() > 1e-8:
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)

    if vision.std() > 1e-8:
        vision = (vision - vision.mean()) / (vision.std() + 1e-8)

    return audio.astype(np.float32), vision.astype(np.float32)


# ── High-level extraction for a single video ─────────────────────

def extract_from_video(
    video_path: str,
    seq_len:    int = SEQ_LEN,
    audio_dim:  int = AUDIO_DIM,
    vision_dim: int = VISION_DIM,
) -> dict:
    """
    Full feature extraction pipeline for one video file.

    Returns:
        {
          'audio':      np.ndarray (seq_len, 768)  normalized
          'vision':     np.ndarray (seq_len, 512)  normalized
          'audio_raw':  np.ndarray — before normalization
          'vision_raw': np.ndarray — before normalization
        }
    """
    audio_raw  = extract_audio_features(video_path, seq_len, audio_dim)
    vision_raw = extract_vision_features(video_path, seq_len, vision_dim)

    audio_norm, vision_norm = apply_normalization(
        audio_raw.copy(),
        vision_raw.copy(),
    )

    return {
        'audio':      audio_norm,
        'vision':     vision_norm,
        'audio_raw':  audio_raw,
        'vision_raw': vision_raw,
    }