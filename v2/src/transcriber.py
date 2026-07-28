"""
Base transcriber. Thin wrapper around a pluggable ASR backend.

After the diarization + VAD removal, this class does almost nothing beyond
holding an ASR backend reference. Kept for `BatchTranscriber` to subclass.
"""

import gc
import os
import warnings
import torch

warnings.filterwarnings('ignore')
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'

from .config import get_config
from .asr_backends import build_asr_backend
from .logger import get_logger

log = get_logger(__name__)


def is_likely_hallucination(text: str, duration: float,
                            max_words_for_long_audio: int = 2,
                            long_audio_threshold: float = 1.5) -> bool:
    """
    Heuristic to drop ASR hallucinations from non-speech audio.

    Modern ASR decoders (Qwen3-ASR, Whisper, etc.) produce *some* text from
    almost any input — even pure noise — typically a single word or two.
    If a >1.5s audio segment produces ≤2 words, it's almost always a
    hallucination rather than real speech.
    """
    if not text:
        return False
    if duration < long_audio_threshold:
        return False
    word_count = len([w for w in text.strip().split() if w])
    return word_count <= max_words_for_long_audio


class BaseTranscriber:
    """Holds an ASR backend. No VAD, no diarization — just transcription."""

    def __init__(self, nemo_model=None, config=None, use_realtime_model=False,
                 asr_backend=None):
        self.config = config or get_config()

        if asr_backend is None:
            asr_backend = build_asr_backend(
                self.config,
                use_realtime_model=use_realtime_model,
                model_override=nemo_model,
            )
        self.asr_backend = asr_backend
        self.device = asr_backend.device

        backend_label = {
            "nemo": "NVIDIA NeMo", "qwen": "Qwen3-ASR",
            "gemini": "Gemini", "google-stt": "Google Cloud STT",
            "whisper": "Whisper",
        }.get(self.asr_backend.name, self.asr_backend.name)
        log.info("Transcriber ready — ASR only (%s)", backend_label)
        print(f"[OK] ASR ready ({backend_label})\n")

    def transcribe_segment(self, audio_data):
        return self.asr_backend.transcribe(audio_data)

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
