"""
Per-user Session for the multi-user web server (streaming-first, transcript-only).

Each WebSocket connection gets its own Session. The Session:
- consumes incoming PCM frames from the WebSocket
- runs them through a `VADEndpointer` to detect end-of-utterance moments
- pushes them into an `ASRStream` (streaming-native or chunked-adapter)
- forwards interim and final emissions to the WebSocket client

Diarization has been removed entirely — finals carry no speaker label.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from .asr_backends import ASRBackend
from .asr_stream import ASRStream, StreamingResult, VADEndpointer
from .asr_streams import ChunkedASRStream
from .audio_recorder import RollingMP3Writer, recording_base_path
from .config import get_config
from .logger import get_logger
from .transcriber import is_likely_hallucination

log = get_logger(__name__)

EmissionCallback = Callable[[StreamingResult], Awaitable[None] | None]


@dataclass
class Session:
    """Streaming per-user session, transcript-only.

    Public API used by the server:
        start()                — kick off the background streaming loop
        add_audio(pcm)         — enqueue a PCM frame
        set_asr_backend(asr)   — swap ASR mid-session
        stop()                 — graceful async shutdown
        close()                — final cleanup (transcripts, recording)
    """

    asr_backend: ASRBackend
    model_lock: asyncio.Lock
    on_emission: EmissionCallback | None = None
    config: Any = field(default_factory=get_config)

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    transcript_log: list[StreamingResult] = field(default_factory=list)
    transcript_path: Path | None = field(default=None)

    _started_at: datetime = field(default_factory=datetime.now)
    _transcript_file: Any = field(default=None, init=False, repr=False)
    _recorder: Any = field(default=None, init=False, repr=False)
    _audio_queue: asyncio.Queue = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _stream: ASRStream = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _endpointer: VADEndpointer = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _stream_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    # ── Construction ────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.config.save_transcript:
            self._open_transcript_file()

        if self.config.recording_enabled:
            base = recording_base_path(
                prefix="web",
                asr_backend_name=self.asr_backend.name,
                asr_model_name=self.asr_backend.model_name,
                recordings_dir=self.config.recording_dir,
            )
            self._recorder = RollingMP3Writer(
                base_path=base,
                sample_rate=self.config.sample_rate,
                max_bytes=self.config.recording_max_part_bytes,
                bitrate_kbps=self.config.recording_bitrate_kbps,
                fmt=self.config.recording_format,
            )

        self._audio_queue = asyncio.Queue(maxsize=512)
        self._stream = self._build_stream(self.asr_backend)
        self._endpointer = VADEndpointer(
            sample_rate=self.config.sample_rate,
            silence_threshold=self.config.silence_threshold,
            min_silence_ms=self.config.min_silence_ms,
            min_speech_ms=200,
        )

        log.info("Session %s started (backend=%s, model=%s, stream=%s, recording=%s)",
                 self.session_id,
                 self.asr_backend.name, self.asr_backend.model_name,
                 type(self._stream).__name__,
                 "yes" if self._recorder is not None else "no")

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> asyncio.Task:
        if self._stream_task is not None and not self._stream_task.done():
            return self._stream_task
        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        return self._stream_task

    def add_audio(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if pcm.ndim > 1:
            pcm = pcm.flatten()

        if self._recorder is not None:
            self._recorder.write(pcm)

        try:
            self._audio_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                _ = self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(pcm)
                log.warning("Session %s audio queue full — dropped oldest frame",
                            self.session_id)
            except Exception:
                pass

    def set_asr_backend(self, asr: ASRBackend) -> None:
        prev = (self.asr_backend.name, self.asr_backend.model_name)
        self.asr_backend = asr
        self._endpointer.reset()
        self._stream = self._build_stream(asr)
        log.info("Session %s changed ASR: %s/%s → %s/%s",
                 self.session_id, prev[0], prev[1], asr.name, asr.model_name)

    def close(self) -> None:
        log.info("Session %s end (final results=%d)",
                 self.session_id, len(self.transcript_log))
        self._running = False

        if self._transcript_file is not None:
            try:
                self._transcript_file.write(
                    f"\n{'=' * 60}\n"
                    f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                self._transcript_file.close()
            except Exception:
                log.exception("Session %s transcript close failed", self.session_id)
            self._transcript_file = None

        if self._recorder is not None:
            try:
                self._recorder.close()
                if self._recorder.parts:
                    log.info("Session %s recording: %d part(s) saved",
                             self.session_id, len(self._recorder.parts))
            except Exception:
                log.exception("Session %s recorder close failed", self.session_id)
            self._recorder = None

    async def stop(self) -> None:
        """Graceful shutdown — cancel stream loop, flush utterance, close."""
        self._running = False

        if self._stream_task is not None:
            try:
                self._audio_queue.put_nowait(_STOP_SENTINEL)
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stream_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._stream_task.cancel()
            except Exception:
                log.exception("Session %s stream loop crashed", self.session_id)
            self._stream_task = None

        try:
            finals = await self._stream.end_utterance()
            for f in finals:
                await self._handle_final(f)
        except Exception:
            log.exception("Session %s final flush failed", self.session_id)

        try:
            await self._stream.close()
        except Exception:
            log.exception("Session %s stream close failed", self.session_id)

        self.close()

    # ── Streaming loop ──────────────────────────────────────────────────

    async def _stream_loop(self) -> None:
        try:
            while self._running:
                try:
                    pcm = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                if pcm is _STOP_SENTINEL:
                    break

                # VAD endpoint detection (energy-based; cheap)
                endpoint = self._endpointer.feed(pcm)

                # Feed the streaming ASR
                try:
                    interim_emissions = await self._stream.feed(pcm)
                except Exception:
                    log.exception("Session %s stream.feed failed", self.session_id)
                    interim_emissions = []

                for em in interim_emissions:
                    if em.is_final:
                        await self._handle_final(em)
                        self._endpointer.reset()
                    else:
                        await self._emit(em)

                if endpoint:
                    try:
                        finals = await self._stream.end_utterance()
                    except Exception:
                        log.exception("Session %s stream.end_utterance failed",
                                      self.session_id)
                        finals = []
                    for f in finals:
                        await self._handle_final(f)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Session %s stream_loop crashed", self.session_id)

    async def _handle_final(self, result: StreamingResult) -> None:
        """Filter hallucinations, emit final, persist to transcript file."""
        if not result.text:
            return
        if is_likely_hallucination(result.text, result.duration):
            log.debug("Session %s dropped hallucination final (%.2fs → %r)",
                      self.session_id, result.duration, result.text)
            return
        result.speaker = None
        await self._emit(result)
        self._record_final(result)

    async def _emit(self, result: StreamingResult) -> None:
        if self.on_emission is None:
            return
        cb = self.on_emission(result)
        if asyncio.iscoroutine(cb):
            await cb

    # ── Stream construction ─────────────────────────────────────────────

    def _build_stream(self, backend: ASRBackend) -> ASRStream:
        """Pick the right ASRStream wrapper for the backend.

        Native streaming when supported; chunked adapter otherwise."""
        # NeMo cache-aware streaming
        if (backend.name == "nemo" and
                "streaming" in (backend.model_name or "").lower()):
            try:
                from .asr_streams import NemoStreamingStream, configure_nemo_streaming
                asr_model = getattr(backend, "model", None)
                if asr_model is None:
                    raise RuntimeError("NemoASRBackend has no .model attribute")
                configure_nemo_streaming(asr_model)
                log.info("Using NemoStreamingStream for %s/%s",
                         backend.name, backend.model_name)
                return NemoStreamingStream(
                    asr_model=asr_model,
                    model_name=backend.model_name,
                    device=getattr(backend, "device", "cpu"),
                    sample_rate=self.config.sample_rate,
                    silence_threshold=self.config.silence_threshold,
                    max_utterance_sec=self.config.max_chunk_duration * 5,
                )
            except Exception:
                log.exception("NemoStreamingStream setup failed; falling back")

        # Google Cloud STT v2 streaming
        if backend.name == "google-stt":
            try:
                from .asr_streams import GoogleSTTStream
                log.info("Using GoogleSTTStream for %s/%s",
                         backend.name, backend.model_name)
                return GoogleSTTStream(
                    project_id=self.config.google_stt_project,
                    location=getattr(backend, "location", self.config.google_stt_location),
                    model=backend.model_name,
                    sample_rate=self.config.sample_rate,
                    language_codes=self.config.google_stt_languages,
                    max_utterance_sec=self.config.max_chunk_duration * 5,
                )
            except Exception:
                log.exception("GoogleSTTStream setup failed; falling back")

        # Gemini Live — only true Live API model IDs support bidiGenerateContent.
        # Batch model IDs (e.g. gemini-2.5-flash, gemini-2.5-pro) must fall
        # through to ChunkedASRStream below.
        _live_ids = {
            "gemini-2.0-flash-live-001",
            "gemini-live-2.5-flash-preview",
            "gemini-live-2.0-flash-preview-04-09",
            "gemini-2.0-flash-live",
        }
        _is_live = (
            backend.model_name in _live_ids
            or "live" in backend.model_name.lower()
        )
        if backend.name == "gemini" and _is_live:
            try:
                from .asr_streams import GeminiLiveStream
                log.info("Using GeminiLiveStream for %s/%s",
                         backend.name, backend.model_name)
                return GeminiLiveStream(
                    api_key=self.config.gemini_api_key,
                    model=backend.model_name,
                    sample_rate=self.config.sample_rate,
                    language=self.config.gemini_language,
                    max_utterance_sec=self.config.max_chunk_duration * 5,
                )
            except Exception:
                log.exception("GeminiLiveStream setup failed; falling back")


        # Default: chunked adapter (Qwen, Whisper, NeMo offline, etc.).
        # Per-backend tuning: each ASR family has a sweet-spot window where
        # accuracy is good but ASR latency stays sub-second:
        #
        #   Whisper  — trained on 30s windows; rewards more context.       → 5s
        #   Qwen3-ASR — encoder-decoder LLM, balanced.                     → 3s
        #   NeMo offline (Conformer-CTC) — works well with shorter chunks. → 3s
        #   Gemini batch / other — conservative middle ground.             → 4s
        #
        # interim_interval_ms is driven by config.chunk_duration (default
        # 500ms) so the user can crank it down further globally.
        window_by_backend = {
            "qwen": 3.0,
            "whisper": 5.0,
            "nemo": 3.0,
            "gemini": 4.0,
            "google-stt": 3.0,
        }
        interim_window_sec = window_by_backend.get(
            backend.name, float(self.config.max_chunk_duration)
        )
        log.info("Using ChunkedASRStream for %s/%s "
                 "(interim every %.0fms, window %.1fs)",
                 backend.name, backend.model_name,
                 self.config.chunk_duration * 1000, interim_window_sec)
        return ChunkedASRStream(
            backend=backend,
            sample_rate=self.config.sample_rate,
            interim_interval_ms=int(self.config.chunk_duration * 1000),
            max_utterance_sec=self.config.max_chunk_duration * 5,
            silence_threshold=self.config.silence_threshold,
            interim_window_sec=interim_window_sec,
        )

    # ── Persistence ─────────────────────────────────────────────────────

    def _open_transcript_file(self) -> None:
        timestamp = self._started_at.strftime(self.config.timestamp_format)
        path = self.config.transcript_dir / f"web_{timestamp}.txt"
        counter = 1
        while path.exists():
            counter += 1
            path = self.config.transcript_dir / f"web_{timestamp}_{counter}.txt"

        self.transcript_path = path
        self._transcript_file = open(path, "w", encoding="utf-8")
        f = self._transcript_file
        f.write(f"Web Transcript ({self.asr_backend.name.upper()} ASR)\n")
        f.write(f"Model: {self.asr_backend.model_name}\n")
        f.write(f"Session: {self.session_id}\n")
        f.write(f"Started: {self._started_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

    def _record_final(self, r: StreamingResult) -> None:
        self.transcript_log.append(r)
        if self._transcript_file is not None:
            line = f"[{r.start:.2f}s - {r.start + r.duration:.2f}s] {r.text}"
            try:
                self._transcript_file.write(line + "\n")
                self._transcript_file.flush()
            except Exception:
                log.exception("Session %s transcript write failed", self.session_id)


_STOP_SENTINEL = object()
