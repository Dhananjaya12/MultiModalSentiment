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
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Constants — must match data prep notebook ─────────────────────
SEQ_LEN    = 300
AUDIO_DIM  = 768   # wav2vec2-base hidden size
VISION_DIM = 512   # CLIP ViT-B/32
SR         = 16000
FRAME_STEP       = 3     # training-time sampling interval
MAX_VISION_FRAMES = 32    # production cap; output is still padded to SEQ_LEN
CLIP_BATCH_SIZE   = 16

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


def preload_feature_models(load_audio: bool = True, load_vision: bool = True):
    """Load feature encoders during server startup instead of the first request."""
    started = time.perf_counter()
    if load_audio:
        _get_wav2vec()
    if load_vision:
        _get_clip()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f'[TIMING][startup] feature_models_load={time.perf_counter() - started:.3f}s', flush=True)

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

        with torch.inference_mode():
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


def extract_vision_features(
    video_path,
    seq_len=SEQ_LEN,
    vision_dim=VISION_DIM,
    max_frames=MAX_VISION_FRAMES,
    batch_size=CLIP_BATCH_SIZE,
):
    """Extract uniformly sampled CLIP features using batched GPU inference."""
    from PIL import Image

    try:
        clip_m, clip_prep = _get_clip()
        device = _get_device()
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            sample_count = min(max_frames, total_frames)
            target_indices = set(
                np.linspace(0, total_frames - 1, sample_count, dtype=np.int64).tolist()
            )
        else:
            target_indices = None

        images = []
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            should_sample = (
                frame_index in target_indices
                if target_indices is not None
                else frame_index % FRAME_STEP == 0 and len(images) < max_frames
            )
            if should_sample:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(clip_prep(Image.fromarray(rgb)))
                if len(images) >= max_frames:
                    break
            frame_index += 1

        cap.release()
        if not images:
            return np.zeros((seq_len, vision_dim), dtype=np.float32)

        feature_batches = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                image_batch = torch.stack(images[start:start + batch_size]).to(device)
                encoded = clip_m.encode_image(image_batch)
                feature_batches.append(encoded.float().cpu().numpy())

        features = np.concatenate(feature_batches, axis=0).astype(np.float32)
        if features.shape[1] > vision_dim:
            features = features[:, :vision_dim]
        elif features.shape[1] < vision_dim:
            features = np.pad(features, ((0, 0), (0, vision_dim - features.shape[1])))

        if len(features) < seq_len:
            features = np.vstack([
                features,
                np.zeros((seq_len - len(features), vision_dim), dtype=np.float32),
            ])
        else:
            features = features[:seq_len]

        print(
            f'[TIMING][features] vision_frames_used={len(images)} | '
            f'clip_batches={(len(images) + batch_size - 1) // batch_size}',
            flush=True,
        )
        return features.astype(np.float32)

    except Exception as e:
        print(f'Vision feature extraction failed: {e}', flush=True)
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
    started = time.perf_counter()
    text = text.strip() if text else ''

    if len(text) == 0:
        input_ids      = torch.zeros(max_len, dtype=torch.long)
        attention_mask = torch.ones(max_len,  dtype=torch.long)
        print(f'[TIMING][features] text_tokenization={time.perf_counter() - started:.3f}s', flush=True)
        return input_ids, attention_mask

    enc = tokenizer(
        text,
        max_length     = max_len,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt',
    )
    result = (
        enc['input_ids'].squeeze(0),
        enc['attention_mask'].squeeze(0),
    )
    print(f'[TIMING][features] text_tokenization={time.perf_counter() - started:.3f}s', flush=True)
    return result


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
    max_vision_frames: int = MAX_VISION_FRAMES,
    clip_batch_size: int = CLIP_BATCH_SIZE,
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
    total_started = time.perf_counter()

    started = time.perf_counter()
    audio_raw = extract_audio_features(video_path, seq_len, audio_dim)
    audio_seconds = time.perf_counter() - started

    started = time.perf_counter()
    vision_raw = extract_vision_features(
        video_path, seq_len, vision_dim, max_vision_frames, clip_batch_size
    )
    vision_seconds = time.perf_counter() - started

    started = time.perf_counter()
    audio_norm, vision_norm = apply_normalization(
        audio_raw.copy(),
        vision_raw.copy(),
    )
    normalization_seconds = time.perf_counter() - started

    timings = {
        'audio_features': audio_seconds,
        'vision_features': vision_seconds,
        'normalization': normalization_seconds,
        'feature_extraction_total': time.perf_counter() - total_started,
    }
    print(
        '[TIMING][features] ' + ' | '.join(
            f'{name}={seconds:.3f}s' for name, seconds in timings.items()
        ),
        flush=True,
    )

    return {
        'audio':      audio_norm,
        'vision':     vision_norm,
        'audio_raw':  audio_raw,
        'vision_raw': vision_raw,
        'timings':    timings,
    }