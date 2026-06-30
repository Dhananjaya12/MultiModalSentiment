"""Gemini fallback predictor for deployment-time robustness.

The custom Transformer model remains the primary predictor. This module is only
used when the app decides to fall back because local inference failed or returned
low confidence.
"""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from inference.utils import format_result


_LABEL_TO_SCORE = {
    'negative': -1.0,
    'neutral': 0.0,
    'positive': 1.0,
}

_CONFIDENCE_TO_VALUE = {
    'low': 0.45,
    'medium': 0.65,
    'high': 0.85,
}


class GeminiFallbackPredictor:
    """Small wrapper around Gemini for text/video sentiment fallback."""

    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-flash') -> None:
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _mime_type(path: str) -> str:
        guessed, _ = mimetypes.guess_type(path)
        return guessed or 'application/octet-stream'

    @classmethod
    def _file_part(cls, path: str) -> types.Part:
        return types.Part.from_bytes(
            data=Path(path).read_bytes(),
            mime_type=cls._mime_type(path),
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = (text or '').strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = (label or '').strip().lower()
        if label not in _LABEL_TO_SCORE:
            return 'neutral'
        return label

    @staticmethod
    def _normalize_confidence(confidence: str) -> float:
        confidence = (confidence or '').strip().lower()
        return _CONFIDENCE_TO_VALUE.get(confidence, 0.65)

    def _generate(self, text: str = '', video_path: Optional[str] = None) -> dict:
        prompt = """
You are a multimodal sentiment analysis fallback system.

Analyze the provided text and/or video/audio content. Return only valid JSON with this schema:
{
  "label": "negative" | "neutral" | "positive",
  "confidence": "low" | "medium" | "high",
  "reason": "one short sentence",
  "evidence": "brief cues from text, voice, or visuals"
}

If the input is unclear, use label "neutral" and confidence "low".
""".strip()

        parts: list[object] = [prompt]
        clean_text = (text or '').strip()
        if clean_text:
            parts.append(f'\nText/transcript:\n{clean_text}')
        if video_path:
            parts.append('\nVideo/audio input:')
            parts.append(self._file_part(video_path))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=250,
            ),
        )
        return self._extract_json(response.text or '{}')

    def predict(self, text: str = '', video_path: Optional[str] = None) -> dict:
        """Return a UI-compatible result dict."""
        if not (text or '').strip() and not video_path:
            return {'error': 'empty_input', 'message': 'Please provide text or video.'}

        try:
            parsed = self._generate(text=text, video_path=video_path)
            label_key = self._normalize_label(str(parsed.get('label', 'neutral')))
            confidence = self._normalize_confidence(str(parsed.get('confidence', 'medium')))
            score = _LABEL_TO_SCORE[label_key]

            result = format_result(
                raw_score=score,
                snapped_score=score,
                label=label_key.title(),
                confidence=confidence,
                modality_info={
                    'quality': 'fallback',
                    'has_audio': bool(video_path),
                    'has_vision': bool(video_path),
                    'has_text': bool((text or '').strip()),
                    'warnings': [],
                },
                transcript=text or '',
                dataset='meld',
            )
            result['provider'] = 'gemini_fallback'
            result['reason'] = str(parsed.get('reason', '')).strip()
            result['evidence'] = str(parsed.get('evidence', '')).strip()
            return result
        except Exception as exc:
            return {'error': 'gemini_failed', 'message': str(exc)}