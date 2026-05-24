"""
inference/feature_extractor.py

Feature extraction for inference — MUST match data prep notebook exactly.

Settings that must never change:
  sr          = 16000
  audio_dim   = 74  (exact feature combination below)
  vision_dim  = 1404  (468 landmarks x 3)
  seq_len     = 300
  resize      = (320, 240) before MediaPipe
  min_face_detection_confidence = 0.3
"""

import os
import cv2
import librosa
import numpy as np
import threading
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Constants — must match data prep notebook ─────────────────────
SEQ_LEN    = 300
AUDIO_DIM  = 74
VISION_DIM = 1404   # 468 landmarks x 3 coords
RESIZE_W   = 320
RESIZE_H   = 240
SR         = 16000
MIN_AUDIO_LEN = 4096  # minimum samples before feature extraction

# MediaPipe face landmarker model URL
FACE_LANDMARKER_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)
FACE_LANDMARKER_PATH = Path(__file__).parent / 'face_landmarker.task'


def _ensure_face_landmarker():
    """Download face landmarker model if not present."""
    if not FACE_LANDMARKER_PATH.exists():
        print(f'Downloading face landmarker model...')
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
        print(f'✅ Downloaded to {FACE_LANDMARKER_PATH}')


# ── Thread-local MediaPipe instances ──────────────────────────────
_thread_local = threading.local()


def _get_face_landmarker():
    """
    One MediaPipe FaceLandmarker per thread — thread safe.
    Lazy-initializes on first call per thread.
    """
    if not hasattr(_thread_local, 'face_lm'):
        import mediapipe as mp
        _ensure_face_landmarker()
        BaseOptions        = mp.tasks.BaseOptions
        FaceLandmarker     = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode  = mp.tasks.vision.RunningMode

        opts = FaceLandmarkerOpts(
            base_options=BaseOptions(
                model_asset_path=str(FACE_LANDMARKER_PATH)
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
        )
        _thread_local.face_lm = FaceLandmarker.create_from_options(opts)
    return _thread_local.face_lm


# ── Audio feature extraction ──────────────────────────────────────

def extract_audio_features(
    video_path: str,
    seq_len:    int = SEQ_LEN,
    audio_dim:  int = AUDIO_DIM,
) -> np.ndarray:
    """
    Extract exactly 74-dim audio features from a video or audio file.

    Feature breakdown (must match data prep notebook):
      13 MFCC + 13 delta + 13 delta2
      12 chroma
      10 mel (power_to_db)
       1 zcr
       1 rms
       1 spectral centroid
       1 spectral bandwidth
       1 spectral rolloff
       1 spectral flatness
       1 spectral contrast (n_bands=1, first row only)
       6 mel2 (fmax=4000)
      ─────────────────
      74 total

    Args:
        video_path: path to .mp4, .avi, .mov, .wav, .mp3
        seq_len:    number of time steps (300)
        audio_dim:  feature dimension (74)

    Returns:
        np.ndarray shape (seq_len, audio_dim) float32
        Returns zeros if extraction fails.
    """
    audio_path = str(video_path) + '_infer_audio.wav'
    try:
        # Extract audio track at 16kHz mono
        ret = os.system(
            f'ffmpeg -i "{video_path}" -ac 1 -ar {SR} '
            f'"{audio_path}" -y -loglevel quiet'
        )

        if not os.path.exists(audio_path):
            return np.zeros((seq_len, audio_dim), dtype=np.float32)

        y, sr = librosa.load(audio_path, sr=SR)

        # Pad very short clips
        if len(y) < MIN_AUDIO_LEN:
            y = np.pad(y, (0, MIN_AUDIO_LEN - len(y)))

        # ── Extract exactly 74 features ──────────────────────────
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)            # 13
        mfcc_d   = librosa.feature.delta(mfcc)                             # 13
        mfcc_d2  = librosa.feature.delta(mfcc, order=2)                    # 13
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)   # 12
        mel      = librosa.power_to_db(
                       librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10)
                   )                                                         # 10
        zcr      = librosa.feature.zero_crossing_rate(y)                   #  1
        rms      = librosa.feature.rms(y=y)                                #  1
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)           #  1
        bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr)          #  1
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)            #  1
        flatness = librosa.feature.spectral_flatness(y=y)                  #  1
        contrast = librosa.feature.spectral_contrast(
                       y=y, sr=sr, n_bands=1
                   )[:1]                                                    #  1
        mel2     = librosa.power_to_db(
                       librosa.feature.melspectrogram(y=y, sr=sr, n_mels=6,
                                                      fmax=4000)
                   )                                                         #  6
        # Total: 13+13+13+12+10+1+1+1+1+1+1+1+6 = 74 ✅

        features = np.vstack([
            mfcc, mfcc_d, mfcc_d2, chroma, mel,
            zcr, rms, centroid, bw, rolloff, flatness, contrast, mel2
        ]).T  # (T, 74)

        # Pad/truncate time dimension
        T = features.shape[0]
        if T < seq_len:
            features = np.vstack([
                features,
                np.zeros((seq_len - T, audio_dim), dtype=np.float32)
            ])
        else:
            features = features[:seq_len]

        return features.astype(np.float32)

    except Exception as e:
        return np.zeros((seq_len, audio_dim), dtype=np.float32)

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# ── Vision feature extraction ─────────────────────────────────────

def extract_vision_features(
    video_path:  str,
    seq_len:     int = SEQ_LEN,
    vision_dim:  int = VISION_DIM,
) -> np.ndarray:
    """
    Extract 1404-dim face landmark features from every frame of a video.

    Pipeline (must match data prep notebook):
      1. Read frame with OpenCV
      2. Resize to (320, 240) — speeds up MediaPipe, same landmarks
      3. Convert BGR → RGB
      4. Run MediaPipe FaceLandmarker
      5. Flatten 468 landmarks × 3 coords = 1404
      6. Force exactly 1404 dims (truncate if > 1404, pad if < 1404)
      7. Stack all frames → (T, 1404)
      8. Pad/truncate to seq_len=300

    Args:
        video_path: path to video file
        seq_len:    number of time steps (300)
        vision_dim: landmark dimension (1404)

    Returns:
        np.ndarray shape (seq_len, vision_dim) float32
        Returns zeros if no face or extraction fails.
    """
    import mediapipe as mp

    try:
        face_lm = _get_face_landmarker()
        cap     = cv2.VideoCapture(str(video_path))
        frames  = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for faster MediaPipe — same as data prep
            frame  = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = face_lm.detect(mp_img)

            if result.face_landmarks:
                lm     = result.face_landmarks[0]
                coords = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
            else:
                coords = np.zeros(vision_dim, dtype=np.float32)

            # Force exactly vision_dim — same as data prep
            if len(coords) >= vision_dim:
                coords = coords[:vision_dim]
            else:
                coords = np.pad(coords, (0, vision_dim - len(coords)))

            frames.append(coords.astype(np.float32))

        cap.release()

        if len(frames) == 0:
            return np.zeros((seq_len, vision_dim), dtype=np.float32)

        # np.stack — same as data prep (safe when all frames same size)
        features = np.stack(frames, axis=0)  # (T, 1404)

        # Pad/truncate
        T = features.shape[0]
        if T < seq_len:
            features = np.vstack([
                features,
                np.zeros((seq_len - T, vision_dim), dtype=np.float32)
            ])
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

    Args:
        text:      raw text string (may be empty)
        tokenizer: RobertaTokenizerFast instance
        max_len:   maximum token length (128)

    Returns:
        input_ids:      torch.Tensor shape (max_len,) dtype=long
        attention_mask: torch.Tensor shape (max_len,) dtype=long

    Edge cases:
        Empty string → zero input_ids, ones attention_mask
        (ones mask to avoid transformer crash on all-padding input)
    """
    import torch

    text = text.strip() if text else ''

    if len(text) == 0:
        # Empty text — return zeros for ids, ones for mask
        # Ones mask prevents transformer from crashing on all-padding
        input_ids      = torch.zeros(max_len, dtype=torch.long)
        attention_mask = torch.ones(max_len,  dtype=torch.long)
        return input_ids, attention_mask

    enc = tokenizer(
        text,
        max_length      = max_len,
        padding         = 'max_length',
        truncation      = True,
        return_tensors  = 'pt',
    )
    return (
        enc['input_ids'].squeeze(0),       # (max_len,)
        enc['attention_mask'].squeeze(0),  # (max_len,)
    )


# ── Normalization — must match dataloader.py ──────────────────────

def apply_normalization(
    audio:  np.ndarray,
    vision: np.ndarray,
) -> tuple:
    """
    Per-sample normalization — MUST match dataloader.py exactly.

    dataloader.py does:
        audio  = (audio  - audio.mean())  / (audio.std()  + 1e-8)
        vision = (vision - vision.mean()) / (vision.std() + 1e-8)
    """
    audio  = (audio  - audio.mean())  / (audio.std()  + 1e-8)
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

    Args:
        video_path: path to video or audio file

    Returns:
        {
          'audio':      np.ndarray (seq_len, audio_dim)  normalized
          'vision':     np.ndarray (seq_len, vision_dim) normalized
          'audio_raw':  np.ndarray — before normalization (for quality check)
          'vision_raw': np.ndarray — before normalization (for quality check)
        }
    """
    audio_raw  = extract_audio_features(video_path, seq_len, audio_dim)
    vision_raw = extract_vision_features(video_path, seq_len, vision_dim)

    audio_norm, vision_norm = apply_normalization(
        audio_raw.copy(), vision_raw.copy()
    )

    return {
        'audio':      audio_norm,
        'vision':     vision_norm,
        'audio_raw':  audio_raw,
        'vision_raw': vision_raw,
    }