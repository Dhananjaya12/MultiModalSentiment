"""
inference/utils.py

Shared utility functions for inference pipeline.
Mirrors snap_to_valid from training/trainer.py — must stay in sync.
"""

import numpy as np


# ── Label value sets ──────────────────────────────────────────────
MELD_LABEL_VALUES = np.array([-1., 0., 1.], dtype=np.float32)

MOSEI_LABEL_VALUES = np.array([
    -3., -2.6666667, -2.3333333, -2., -1.6666666, -1.3333334,
    -1., -0.6666667, -0.5, -0.33333334, -0.16666667, 0.,
    0.16666667, 0.33333334, 0.5, 0.6666667, 0.8333333, 1.,
    1.1666666, 1.3333334, 1.5, 1.6666666, 1.8333334, 2.,
    2.3333333, 2.6666667, 3.
], dtype=np.float32)


def snap_to_valid(score: float, dataset: str = 'meld') -> float:
    """
    Snap a continuous score to the nearest valid label value.
    Must match trainer.py snap_to_valid exactly.

    Args:
        score:   raw model output (float)
        dataset: 'meld' or 'mosei'

    Returns:
        Nearest valid label value
    """
    label_vals = MELD_LABEL_VALUES if dataset == 'meld' else MOSEI_LABEL_VALUES
    arr        = np.array([score], dtype=np.float32)
    diffs      = np.abs(arr[:, None] - label_vals[None, :])
    return float(label_vals[diffs.argmin(axis=1)[0]])


def score_to_label(score: float) -> str:
    """
    Convert snapped score to human-readable label.

    MELD:  -1 → Negative, 0 → Neutral, 1 → Positive
    MOSEI: ≤ -1 → Negative, -1 to 0 → Slightly Negative, etc.
    """
    if score >= 0.5:
        return 'Positive'
    elif score <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'


def score_to_color(score: float) -> str:
    """Return hex color for a sentiment score."""
    if score >= 0.5:
        return '#00C851'   # green
    elif score <= -0.5:
        return '#FF4444'   # red
    else:
        return '#FFBB33'   # amber


def score_to_emoji(score: float) -> str:
    """Return emoji for a sentiment score."""
    if score >= 0.5:
        return '😊'
    elif score <= -0.5:
        return '😞'
    else:
        return '😐'


def estimate_confidence(raw_score: float, dataset: str = 'meld') -> float:
    """
    Estimate prediction confidence as distance from the nearest snap boundary.

    For MELD with labels [-1, 0, 1]:
      boundaries are at -0.5 and +0.5
      score of 0.9 → 0.4 from boundary → high confidence
      score of 0.1 → 0.4 from boundary → but near neutral → medium

    Returns: 0.0 (low) to 1.0 (high)
    """
    label_vals = MELD_LABEL_VALUES if dataset == 'meld' else MOSEI_LABEL_VALUES
    dists      = np.abs(raw_score - label_vals)
    sorted_d   = np.sort(dists)
    # Distance between closest and second-closest label
    margin     = float(sorted_d[1] - sorted_d[0])
    # Normalize by half the label spacing
    spacing    = float(np.min(np.diff(label_vals))) / 2.0
    confidence = np.clip(margin / spacing, 0.0, 1.0)
    return float(confidence)


def check_modality_quality(
    audio_np:   np.ndarray,
    vision_np:  np.ndarray,
    text_str:   str,
    std_thresh: float = 0.01,
) -> dict:
    """
    Check which modalities have real content vs zeros/silence.

    Returns:
        {
          'quality':    'high' | 'medium' | 'low' | 'none',
          'has_audio':  bool,
          'has_vision': bool,
          'has_text':   bool,
          'active':     int (0-3),
          'warnings':   list[str],
        }
    """
    has_audio  = bool(audio_np.std()  > std_thresh)
    has_vision = bool(vision_np.std() > std_thresh)
    has_text   = bool(len(text_str.strip()) > 0)

    active   = sum([has_audio, has_vision, has_text])
    warnings = []

    if not has_audio:
        warnings.append('No audio detected — audio modality set to zeros')
    if not has_vision:
        warnings.append('No face detected — vision modality set to zeros')
    if not has_text:
        warnings.append('No speech transcribed — text modality empty')

    if active == 3:
        quality = 'high'
    elif active == 2:
        quality = 'medium'
    elif active == 1:
        quality = 'low'
    else:
        quality = 'none'

    return {
        'quality':    quality,
        'has_audio':  has_audio,
        'has_vision': has_vision,
        'has_text':   has_text,
        'active':     active,
        'warnings':   warnings,
    }


def format_result(
    raw_score:       float,
    snapped_score:   float,
    label:           str,
    confidence:      float,
    modality_info:   dict,
    transcript:      str = '',
    dataset:         str = 'meld',
) -> dict:
    """
    Format final result dict for UI consumption.
    """
    return {
        'score':          snapped_score,
        'raw_score':      raw_score,
        'label':          label,
        'confidence':     round(confidence * 100, 1),   # as percentage
        'color':          score_to_color(snapped_score),
        'emoji':          score_to_emoji(snapped_score),
        'transcript':     transcript,
        'modality_quality': modality_info.get('quality', 'unknown'),
        'has_audio':      modality_info.get('has_audio', False),
        'has_vision':     modality_info.get('has_vision', False),
        'has_text':       modality_info.get('has_text', False),
        'warnings':       modality_info.get('warnings', []),
        'dataset':        dataset,
    }