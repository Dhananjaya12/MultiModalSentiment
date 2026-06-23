"""
inference/predictor.py

SentimentPredictor — loads model once, handles all inference modes.

Whisper strategy:
  webcam mode  → whisper-tiny  (fast, ~150MB)
  upload mode  → whisper-base  (accurate, ~140MB)
  Both loaded lazily — only when first used.
"""

import os
import json
import torch
import numpy as np
import time
from pathlib import Path
from typing import Optional

from inference.feature_extractor import (
    extract_from_video,
    extract_text_features,
    apply_normalization,
    preload_feature_models,
    SEQ_LEN, AUDIO_DIM, VISION_DIM,
)
from inference.utils import (
    score_to_label,
    score_to_color,
    score_to_emoji,
    check_modality_quality,
    format_result,
)


class SentimentPredictor:
    """
    Single entry point for all sentiment inference modes.

    Usage:
        predictor = SentimentPredictor(
            model_path  = 'best_model.pt',
            config_path = 'config.json',
        )
        result = predictor.predict_from_text("I love this!")
        result = predictor.predict_from_video("clip.mp4")
    """

    def __init__(
        self,
        model_path:  str,
        config_path: str,
        device:      Optional[str] = None,
    ):
        startup_started = time.perf_counter()
        self.model_path  = Path(model_path)
        self.config_path = Path(config_path)

        # ── Device ────────────────────────────────────────────────
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f'SentimentPredictor: device={self.device}')

        # ── Config ────────────────────────────────────────────────
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.dataset    = self.cfg.get('dataset', 'meld')
        self.max_text_len = self.cfg.get('max_text_len', 128)
        self.max_vision_frames = self.cfg.get('max_vision_frames', 32)
        self.clip_batch_size = self.cfg.get('clip_batch_size', 16)
        self.preload_models = self.cfg.get('preload_inference_models', True)

        # ── Load model ────────────────────────────────────────────
        print('Loading model...')
        model_started = time.perf_counter()
        from model.model import TransformerFusionModel
        self.model = TransformerFusionModel(self.cfg)
        state = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        # Support both raw state dict and checkpoint dict
        if isinstance(state, dict) and 'model_state' in state:
            state = state['model_state']
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._sync_cuda()
        print(f'✅ Model loaded from {model_path}')
        print(f'[TIMING][startup] fusion_model_load={time.perf_counter() - model_started:.3f}s', flush=True)

        # ── Tokenizer ─────────────────────────────────────────────
        print('Loading tokenizer...')
        tokenizer_started = time.perf_counter()
        from transformers import RobertaTokenizerFast
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        print('✅ Tokenizer ready')
        print(f'[TIMING][startup] tokenizer_load={time.perf_counter() - tokenizer_started:.3f}s', flush=True)

        # ── Whisper — lazy loaded ─────────────────────────────────
        self._whisper_tiny = None   # for webcam
        self._whisper_base = None   # for video upload

        if self.preload_models:
            preload_feature_models(load_audio=True, load_vision=True)
            self._get_whisper('upload')
            self._warm_up_model()

        print(f'[TIMING][startup] predictor_total={time.perf_counter() - startup_started:.3f}s', flush=True)

    def _sync_cuda(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _log_timings(scope: str, timings: dict):
        values = ' | '.join(f'{name}={seconds:.3f}s' for name, seconds in timings.items())
        print(f'[TIMING][{scope}] {values}', flush=True)

    def _warm_up_model(self):
        started = time.perf_counter()
        ids = torch.zeros((1, self.max_text_len), dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        audio = torch.zeros((1, SEQ_LEN, AUDIO_DIM), device=self.device)
        vision = torch.zeros((1, SEQ_LEN, VISION_DIM), device=self.device)
        with torch.inference_mode():
            self.model(ids, mask, audio, vision)
        self._sync_cuda()
        print(f'[TIMING][startup] fusion_warmup={time.perf_counter() - started:.3f}s', flush=True)

    # ── Whisper lazy loading ──────────────────────────────────────

    def _get_whisper(self, mode: str = 'upload'):
        """
        Load Whisper model on first use.
        webcam → tiny (fast)
        upload → base (accurate)
        """
        import whisper
        if mode == 'webcam':
            if self._whisper_tiny is None:
                started = time.perf_counter()
                print('Loading whisper-tiny for webcam...')
                self._whisper_tiny = whisper.load_model('tiny')
                self._sync_cuda()
                print(f'[TIMING][startup] whisper_tiny_load={time.perf_counter() - started:.3f}s', flush=True)
            return self._whisper_tiny
        else:
            if self._whisper_base is None:
                started = time.perf_counter()
                print('Loading whisper-base for video upload...')
                self._whisper_base = whisper.load_model('base')
                self._sync_cuda()
                print(f'[TIMING][startup] whisper_base_load={time.perf_counter() - started:.3f}s', flush=True)
            return self._whisper_base

    def _transcribe(self, audio_path: str, mode: str = 'upload') -> tuple:
        """
        Transcribe audio file.

        Returns:
            text:     full transcript string
            segments: list of {start, end, text} dicts
        """
        try:
            total_started = time.perf_counter()
            model_started = time.perf_counter()
            wmodel = self._get_whisper(mode)
            model_seconds = time.perf_counter() - model_started
            self._sync_cuda()
            transcribe_started = time.perf_counter()
            result = wmodel.transcribe(
                audio_path,
                word_timestamps = False,
                language        = 'en',
            )
            self._sync_cuda()
            text = result.get('text', '').strip()
            segments = result.get('segments', [])
            self._log_timings('whisper', {
                'model_get': model_seconds,
                'transcribe': time.perf_counter() - transcribe_started,
                'total': time.perf_counter() - total_started,
            })
            return text, segments
        except Exception as e:
            print(f'Whisper transcription failed: {e}', flush=True)
            return '', []

    # ── Core inference ────────────────────────────────────────────

    # class index → MELD sentiment score
    _CLASS_TO_SCORE = {0: -1.0, 1: 0.0, 2: 1.0}

    def _run_model(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        audio:          np.ndarray,
        vision:         np.ndarray,
    ) -> tuple:
        """
        Single forward pass.
        Returns (class_idx: int, probs: np.ndarray) where probs has shape (3,).
        Classes: 0=negative, 1=neutral, 2=positive
        """
        total_started = time.perf_counter()
        transfer_started = time.perf_counter()
        audio_t  = torch.tensor(audio,  dtype=torch.float32).unsqueeze(0).to(self.device)
        vision_t = torch.tensor(vision, dtype=torch.float32).unsqueeze(0).to(self.device)
        ids_t    = input_ids.unsqueeze(0).to(self.device)
        mask_t   = attention_mask.unsqueeze(0).to(self.device)
        self._sync_cuda()
        transfer_seconds = time.perf_counter() - transfer_started

        self._sync_cuda()
        forward_started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(ids_t, mask_t, audio_t, vision_t)  # (1, 3)
        self._sync_cuda()
        forward_seconds = time.perf_counter() - forward_started

        probs     = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (3,)
        class_idx = int(probs.argmax())
        self._log_timings('model', {
            'tensor_transfer': transfer_seconds,
            'fusion_forward': forward_seconds,
            'total': time.perf_counter() - total_started,
        })
        return class_idx, probs

    # ── Public API ────────────────────────────────────────────────

    def predict_from_text(self, text: str) -> dict:
        """
        Predict sentiment from text only.
        Audio and vision are set to zeros.

        Args:
            text: input string

        Returns:
            result dict from format_result()
        """
        if not text or not text.strip():
            return {'error': 'empty_text', 'message': 'Please provide some text.'}

        input_ids, attention_mask = extract_text_features(
            text, self.tokenizer, self.max_text_len
        )

        audio  = np.zeros((SEQ_LEN, AUDIO_DIM),  dtype=np.float32)
        vision = np.zeros((SEQ_LEN, VISION_DIM), dtype=np.float32)

        class_idx, probs = self._run_model(input_ids, attention_mask, audio, vision)
        snapped      = self._CLASS_TO_SCORE[class_idx]
        label        = score_to_label(snapped)
        confidence   = float(probs[class_idx])

        modality_info = check_modality_quality(audio, vision, text)

        return format_result(
            raw_score     = snapped,
            snapped_score = snapped,
            label         = label,
            confidence    = confidence,
            modality_info = modality_info,
            transcript    = text,
            dataset       = self.dataset,
        )

    def predict_from_video(
        self,
        video_path: str,
        mode:       str = 'upload',
    ) -> dict:
        """
        Predict sentiment from a video/audio file.
        Extracts audio, vision, and transcribes speech.

        Args:
            video_path: path to .mp4 / .mov / .avi / .wav / .mp3
            mode:       'upload' or 'webcam'

        Returns:
            result dict from format_result(), or {'error': ...}
        """
        if not os.path.exists(video_path):
            return {'error': 'file_not_found', 'message': f'File not found: {video_path}'}

        try:
            total_started = time.perf_counter()
            # ── Extract audio + vision ────────────────────────────
            feats = extract_from_video(
                video_path,
                max_vision_frames=self.max_vision_frames,
                clip_batch_size=self.clip_batch_size,
            )
            audio_norm = feats['audio']
            vision_norm= feats['vision']
            audio_raw  = feats['audio_raw']
            vision_raw = feats['vision_raw']

            # ── Transcribe ────────────────────────────────────────
            # Extract audio to temp wav for Whisper
            temp_wav = str(video_path) + '_whisper.wav'
            os.system(
                f'ffmpeg -i "{video_path}" -ac 1 -ar 16000 '
                f'"{temp_wav}" -y -loglevel quiet'
            )
            text, _ = self._transcribe(temp_wav, mode=mode)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

            # ── Modality quality check ────────────────────────────
            modality_info = check_modality_quality(audio_raw, vision_raw, text)

            if modality_info['quality'] == 'none':
                return {
                    'error':   'no_content',
                    'message': 'No analyzable content found. '
                               'Please provide a video with speech or visible face.',
                }

            # ── Tokenize text ─────────────────────────────────────
            input_ids, attention_mask = extract_text_features(
                text, self.tokenizer, self.max_text_len
            )

            # ── Run model ─────────────────────────────────────────
            class_idx, probs = self._run_model(
                input_ids, attention_mask, audio_norm, vision_norm
            )
            snapped    = self._CLASS_TO_SCORE[class_idx]
            label      = score_to_label(snapped)
            confidence = float(probs[class_idx])

            result = format_result(
                raw_score     = snapped,
                snapped_score = snapped,
                label         = label,
                confidence    = confidence,
                modality_info = modality_info,
                transcript    = text,
                dataset       = self.dataset,
            )
            result['timings'] = {
                **feats.get('timings', {}),
                'request_total': time.perf_counter() - total_started,
            }
            return result

        except Exception as e:
            return {'error': 'inference_failed', 'message': str(e)}

    def predict_utterances(
        self,
        video_path: str,
        mode:       str = 'upload',
    ) -> list:
        """
        Predict sentiment per utterance for timeline view.
        Uses Whisper segment timestamps to split video into utterances.

        Args:
            video_path: path to video file
            mode:       'upload' or 'webcam'

        Returns:
            list of dicts:
            [
              {
                'start':      float (seconds),
                'end':        float (seconds),
                'text':       str,
                'score':      float,
                'label':      str,
                'confidence': float,
                'color':      str,
              },
              ...
            ]
        """
        import tempfile

        if not os.path.exists(video_path):
            return []

        try:
            # Transcribe with timestamps
            temp_wav = str(video_path) + '_whisper_seg.wav'
            os.system(
                f'ffmpeg -i "{video_path}" -ac 1 -ar 16000 '
                f'"{temp_wav}" -y -loglevel quiet'
            )
            _, segments = self._transcribe(temp_wav, mode=mode)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

            if not segments:
                # No segments — treat whole video as one utterance
                result = self.predict_from_video(video_path, mode=mode)
                if 'error' not in result:
                    return [{
                        'start':      0.0,
                        'end':        0.0,
                        'text':       result.get('transcript', ''),
                        'score':      result['score'],
                        'label':      result['label'],
                        'confidence': result['confidence'],
                        'color':      result['color'],
                        'emoji':      result['emoji'],
                    }]
                return []

            results = []
            for seg in segments:
                start = seg.get('start', 0.0)
                end   = seg.get('end',   0.0)
                text  = seg.get('text',  '').strip()

                if not text:
                    continue

                # Extract clip for this segment
                clip_path = str(video_path) + f'_seg_{start:.1f}.mp4'
                try:
                    duration = max(end - start, 0.5)  # min 0.5s
                    os.system(
                        f'ffmpeg -i "{video_path}" '
                        f'-ss {start:.2f} -t {duration:.2f} '
                        f'-c copy "{clip_path}" -y -loglevel quiet'
                    )

                    if os.path.exists(clip_path):
                        feats = extract_from_video(clip_path)
                        input_ids, attention_mask = extract_text_features(
                            text, self.tokenizer, self.max_text_len
                        )
                        class_idx, probs = self._run_model(
                            input_ids, attention_mask,
                            feats['audio'], feats['vision']
                        )
                        snapped    = self._CLASS_TO_SCORE[class_idx]
                        label      = score_to_label(snapped)
                        confidence = float(probs[class_idx])

                        results.append({
                            'start':      round(start, 2),
                            'end':        round(end, 2),
                            'text':       text,
                            'score':      snapped,
                            'raw_score':  snapped,
                            'label':      label,
                            'confidence': round(confidence * 100, 1),
                            'color':      score_to_color(snapped),
                            'emoji':      score_to_emoji(snapped),
                        })
                finally:
                    if os.path.exists(clip_path):
                        os.remove(clip_path)

            return results

        except Exception as e:
            print(f'predict_utterances error: {e}')
            return []