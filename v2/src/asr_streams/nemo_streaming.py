"""
NemoStreamingStream — true streaming ASR using NeMo's cache-aware
FastConformer Hybrid model.

The model (`stt_en_fastconformer_hybrid_large_streaming_80ms`) is a
Conformer-Transducer that supports incremental inference: each call to
`conformer_stream_step()` extends the encoder/decoder cache with a new
chunk of audio features and emits any tokens that became confident.

Latency floor: ~80 ms of right-context lookahead (model property).
Compared to ChunkedASRStream:
- Each audio sample is encoded ONCE (no re-encoding the rolling buffer)
- Tokens emit continuously as confidence is reached, not in chunks
- Compute roughly proportional to audio duration, not to interim cadence

Implementation notes:
- We accumulate raw PCM into a `chunk_size_sec` window (typically ~80ms),
  preprocess to mel features once, and feed to the model's stream step.
- The model stack: feature -> encoder (cache-aware) -> decoder (RNN-T).
  We hold three caches: `cache_last_channel`, `cache_last_time`,
  `cache_last_channel_len`. They're initialized via
  `model.encoder.get_initial_cache_state()`.
- Decoded tokens come back as a list of hypotheses; we read their `.text`
  and detect "new text" by treating the running text as monotonically
  growing.

This is the production-style "streaming" replaces the chunked-adapter for
NeMo backends.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..asr_stream import StreamingResult, new_utterance_id, now_ms
from ..audio_utils import detect_silence
from ..logger import get_logger

log = get_logger(__name__)


@dataclass
class NemoStreamingStream:
    """True streaming wrapper for NeMo cache-aware Conformer models."""

    asr_model: Any  # nemo_asr.models.ASRModel — already loaded
    model_name: str
    device: str = "cpu"  # 'cuda' | 'mps' | 'cpu'
    sample_rate: int = 16000
    silence_threshold: float = 0.004
    # Chunk size for one streaming step. NeMo's 80ms model uses ~80ms chunks
    # internally; we feed in those-sized blocks.
    chunk_size_sec: float = 0.16  # 160ms is a safe multiple of frame size
    # Max utterance before forcing a final
    max_utterance_sec: float = 30.0

    name: str = "nemo-streaming"

    # Per-utterance state
    _pcm_pending: list[float] = field(default_factory=list, init=False, repr=False)
    _utterance_id: str = field(default="", init=False)
    _utterance_start_offset: float = field(default=0.0, init=False)
    _absolute_offset: float = field(default=0.0, init=False)
    _last_emitted_text: str = field(default="", init=False)
    _utterance_samples: int = field(default=0, init=False)

    # Cache-aware streaming state — owned per stream (per-session)
    _cache_last_channel: Any = field(default=None, init=False, repr=False)
    _cache_last_time: Any = field(default=None, init=False, repr=False)
    _cache_last_channel_len: Any = field(default=None, init=False, repr=False)
    _previous_hypotheses: Any = field(default=None, init=False, repr=False)
    _pred_out: Any = field(default=None, init=False, repr=False)

    # Per-stream lock — torch operations on the cache aren't reentrant
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = "nemo-streaming"
        self._reset_state()
        log.info("NemoStreamingStream initialized: model=%s device=%s chunk=%.0fms",
                 self.model_name, self.device, self.chunk_size_sec * 1000)

    # ── Public protocol ─────────────────────────────────────────────────

    async def feed(self, pcm: np.ndarray) -> list[StreamingResult]:
        """Push PCM frames into the stream. Returns a list of emissions
        (interim hypotheses if new text is decoded, finals when a force-
        flush at max_utterance_sec triggers)."""
        if pcm.size == 0:
            return []
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if pcm.ndim > 1:
            pcm = pcm.flatten()

        self._pcm_pending.extend(pcm.tolist())
        self._utterance_samples += pcm.size

        emissions: list[StreamingResult] = []

        # Force-flush if utterance got too long
        utterance_dur = self._utterance_samples / float(self.sample_rate)
        if utterance_dur >= self.max_utterance_sec:
            log.info("NeMo stream %s force-flushing at %.1fs",
                     self._utterance_id, utterance_dur)
            emissions.extend(await self.end_utterance())
            return emissions

        # Drain the pending PCM buffer in chunk_size_sec slices, feeding each
        # to the streaming model. Each call may produce one interim emission
        # if new text was decoded.
        chunk_samples = int(self.chunk_size_sec * self.sample_rate)
        loop = asyncio.get_running_loop()
        while len(self._pcm_pending) >= chunk_samples:
            chunk = np.asarray(self._pcm_pending[:chunk_samples], dtype=np.float32)
            del self._pcm_pending[:chunk_samples]

            # Skip if this chunk is silent — saves an inference call AND
            # avoids drift on the cache from feeding noise.
            if detect_silence(chunk, self.silence_threshold):
                continue

            try:
                text = await loop.run_in_executor(None, self._stream_step, chunk)
            except Exception:
                log.exception("NeMo stream %s step failed", self._utterance_id)
                continue

            if text and text != self._last_emitted_text:
                self._last_emitted_text = text
                emissions.append(StreamingResult(
                    id=self._utterance_id,
                    text=text,
                    is_final=False,
                    start=self._utterance_start_offset,
                    duration=self._utterance_samples / float(self.sample_rate),
                ))

        return emissions

    async def end_utterance(self) -> list[StreamingResult]:
        """Caller signals utterance end (e.g., VAD endpoint). Emit a final,
        clear caches so the next utterance starts with a clean state."""
        if self._utterance_samples == 0:
            return []

        # Flush any leftover PCM (less than a full chunk) by padding to a
        # full chunk and running a final step.
        loop = asyncio.get_running_loop()
        if self._pcm_pending:
            chunk_samples = int(self.chunk_size_sec * self.sample_rate)
            pad_n = chunk_samples - len(self._pcm_pending)
            if pad_n > 0:
                chunk = np.concatenate([
                    np.asarray(self._pcm_pending, dtype=np.float32),
                    np.zeros(pad_n, dtype=np.float32),
                ])
            else:
                chunk = np.asarray(self._pcm_pending[:chunk_samples], dtype=np.float32)
            self._pcm_pending = []
            try:
                text = await loop.run_in_executor(None, self._stream_step, chunk)
                if text:
                    self._last_emitted_text = text
            except Exception:
                log.exception("NeMo stream %s final step failed", self._utterance_id)

        text = self._last_emitted_text or ""
        utterance_dur = self._utterance_samples / float(self.sample_rate)
        result = StreamingResult(
            id=self._utterance_id,
            text=text,
            is_final=True,
            start=self._utterance_start_offset,
            duration=utterance_dur,
        )

        # Advance the timeline and reset everything for next utterance
        self._absolute_offset = self._utterance_start_offset + utterance_dur
        self._reset_state()
        return [result] if text else []

    async def close(self) -> None:
        # Nothing held open beyond the model itself (which is shared)
        return None

    # ── Internals ───────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Reset caches + utterance metadata for a fresh utterance."""
        self._pcm_pending = []
        self._utterance_id = new_utterance_id()
        self._utterance_start_offset = self._absolute_offset
        self._utterance_samples = 0
        self._last_emitted_text = ""
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._previous_hypotheses = None
        self._pred_out = None

    def _stream_step(self, pcm_chunk: np.ndarray) -> str:
        """Run one cache-aware streaming inference step on `pcm_chunk`.
        Returns the (possibly extended) running transcript text.

        Holds `self._lock` so multiple coroutines can't race on the cache.
        """
        with self._lock:
            # Convert to a length-1 batch
            audio_signal = torch.from_numpy(pcm_chunk).float().unsqueeze(0)
            audio_length = torch.tensor([pcm_chunk.size], dtype=torch.long)
            if self.device == "cuda":
                audio_signal = audio_signal.cuda()
                audio_length = audio_length.cuda()
            elif self.device == "mps":
                audio_signal = audio_signal.to("mps")
                audio_length = audio_length.to("mps")

            # Initialize caches on first call
            if self._cache_last_channel is None:
                (self._cache_last_channel,
                 self._cache_last_time,
                 self._cache_last_channel_len) = (
                     self.asr_model.encoder.get_initial_cache_state(batch_size=1)
                 )

            with torch.no_grad():
                # 1) Featurize raw audio → mel features
                processed_signal, processed_signal_length = (
                    self.asr_model.preprocessor(
                        input_signal=audio_signal,
                        length=audio_length,
                    )
                )
                # 2) One streaming step (encoder + decoder, cache-aware)
                step_outputs = self.asr_model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out,
                    drop_extra_pre_encoded=None,
                    return_transcription=True,
                )

            # NeMo 2.7 cache-aware stream step returns a 6-tuple:
            #   (transcribed_texts, current_hypotheses,
            #    cache_last_channel_next, cache_last_time_next,
            #    cache_last_channel_len_next, pred_out_stream)
            # transcribed_texts[0] is the running text for batch element 0.
            try:
                (transcribed_texts, current_hypotheses,
                 cache_last_channel_next, cache_last_time_next,
                 cache_last_channel_len_next, pred_out) = step_outputs
            except (TypeError, ValueError):
                log.warning("Unexpected stream-step return shape: %s, len=%s",
                            type(step_outputs).__name__,
                            len(step_outputs) if hasattr(step_outputs, "__len__") else "?")
                return self._last_emitted_text

            # Update cache for next call
            self._cache_last_channel = cache_last_channel_next
            self._cache_last_time = cache_last_time_next
            self._cache_last_channel_len = cache_last_channel_len_next
            self._previous_hypotheses = current_hypotheses
            self._pred_out = pred_out

            # transcribed_texts is typically a list[str] of length=batch_size
            if transcribed_texts and isinstance(transcribed_texts, list):
                t = transcribed_texts[0]
                if isinstance(t, str):
                    return t.strip()
                # Some versions return Hypothesis objects
                if hasattr(t, "text"):
                    return (t.text or "").strip()

            # Fallback: stitch from current_hypotheses
            if current_hypotheses:
                h = current_hypotheses[0] if isinstance(current_hypotheses, list) else current_hypotheses
                if hasattr(h, "text"):
                    return (h.text or "").strip()

            return self._last_emitted_text


def configure_nemo_streaming(asr_model: Any) -> None:
    """Configure a NeMo ASR model for cache-aware streaming inference.

    The 80ms streaming model uses an attention context of [70, 13]
    (left, right) chunks. The default after pretraining is [70, 0] (no
    lookahead), which gives the lowest latency but slightly worse accuracy.
    [70, 13] is the recommended balance per the model card.
    """
    try:
        # Method present on cache-aware Conformer models
        asr_model.encoder.set_default_att_context_size([70, 13])
    except Exception as e:
        log.warning("Could not set att_context_size on model: %s "
                    "(model may not be a cache-aware streaming model)", e)

    # Force single-sequence greedy decoding because batched greedy decoding
    # does not support `partial_hypotheses` which is required for streaming.
    try:
        from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import (
            GreedyBatchedRNNTInfer,
            GreedyRNNTInfer,
        )
        if hasattr(asr_model, 'decoding') and hasattr(asr_model.decoding, 'decoding'):
            current_dec = asr_model.decoding.decoding
            if isinstance(current_dec, GreedyBatchedRNNTInfer):
                log.info("Switching NeMo decoding strategy from GreedyBatchedRNNTInfer to GreedyRNNTInfer to support partial_hypotheses")
                new_dec = GreedyRNNTInfer(
                    decoder_model=current_dec.decoder,
                    joint_model=current_dec.joint,
                    blank_index=current_dec._blank_index,
                    max_symbols_per_step=current_dec.max_symbols,
                    preserve_alignments=current_dec.preserve_alignments,
                    preserve_frame_confidence=current_dec.preserve_frame_confidence,
                    confidence_method_cfg=getattr(current_dec, 'confidence_method_cfg', None),
                )
                asr_model.decoding.decoding = new_dec
                asr_model.decoding.cfg.strategy = 'greedy'
    except Exception as e:
        log.warning("Could not force GreedyRNNTInfer strategy: %s", e)

    asr_model.eval()
