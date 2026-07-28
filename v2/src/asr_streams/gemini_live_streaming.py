"""
GeminiLiveStream — true streaming via Gemini Live API
(`client.aio.live.connect`).

Bidirectional protocol over gRPC: we open a single session per utterance,
push raw 16kHz int16 PCM in, receive text tokens out as they're decoded.
Configured with `response_modalities=["TEXT"]` so we get transcripts (not
audio responses).

Latency: typically 200-500ms first-word; transcription is server-side.

Auth: API key via GEMINI_API_KEY (same as the GeminiASRBackend).

Cost: each audio second is billed. Idle sessions don't incur charges
because we only open the gRPC stream during a speaking phase.

Note: at the time of writing, Gemini Live model availability depends on
the model id. `gemini-live-2.5-flash-preview` and similar live-specific
model ids are required — generic gemini-2.5-flash won't work in live mode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..asr_stream import StreamingResult, new_utterance_id
from ..logger import get_logger

log = get_logger(__name__)


@dataclass
class GeminiLiveStream:
    """Streaming wrapper over Gemini Live API."""

    api_key: str
    model: str = "gemini-live-2.5-flash-preview"
    sample_rate: int = 16000
    language: str | None = "English"
    max_utterance_sec: float = 30.0

    name: str = "gemini-live"
    model_name: str = ""

    _client: Any = field(default=None, init=False, repr=False)
    _session: Any = field(default=None, init=False, repr=False)
    _session_ctx: Any = field(default=None, init=False, repr=False)
    _receive_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _emissions_queue: asyncio.Queue = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _utterance_id: str = field(default="", init=False)
    _utterance_start_offset: float = field(default=0.0, init=False)
    _absolute_offset: float = field(default=0.0, init=False)
    _utterance_samples: int = field(default=0, init=False)
    _running_text: str = field(default="", init=False)
    _stream_active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GeminiLiveStream requires `google-genai`. "
                "Install with: pip install -U google-genai"
            ) from e
        if not self.api_key:
            raise RuntimeError(
                "GeminiLiveStream requires an API key. Set GEMINI_API_KEY."
            )

        from google import genai
        self._client = genai.Client(api_key=self.api_key)
        self.model_name = self.model
        self.name = "gemini-live"
        log.info("GeminiLiveStream ready (model=%s, lang=%s)",
                 self.model, self.language or "auto")
        print(f"[OK] Using Gemini Live (streaming, model: {self.model})")

    # ── Protocol ────────────────────────────────────────────────────────

    async def feed(self, pcm: np.ndarray) -> list[StreamingResult]:
        if pcm.size == 0:
            return []
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if pcm.ndim > 1:
            pcm = pcm.flatten()

        if not self._stream_active:
            await self._start_stream()

        self._utterance_samples += pcm.size

        # Send 16kHz int16 LINEAR16 audio
        int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
        try:
            await self._session.send_realtime_input(
                audio={"data": int16.tobytes(), "mime_type": "audio/pcm;rate=16000"}
            )
        except Exception:
            log.exception("GeminiLiveStream %s send failed", self._utterance_id)
            return []

        # Drain anything the receive task has accumulated
        emissions: list[StreamingResult] = []
        while not self._emissions_queue.empty():
            try:
                emissions.append(self._emissions_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        utterance_dur = self._utterance_samples / float(self.sample_rate)
        if utterance_dur >= self.max_utterance_sec:
            emissions.extend(await self.end_utterance())
        return emissions

    async def end_utterance(self) -> list[StreamingResult]:
        if not self._stream_active:
            return []

        # Tear down the session — closing it flushes any final tokens
        try:
            if self._session_ctx is not None:
                await self._session_ctx.__aexit__(None, None, None)
        except Exception:
            log.exception("GeminiLiveStream %s session close failed",
                          self._utterance_id)
        self._session = None
        self._session_ctx = None

        if self._receive_task is not None:
            try:
                await asyncio.wait_for(self._receive_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            self._receive_task = None

        # Drain any remaining emissions
        emissions: list[StreamingResult] = []
        while not self._emissions_queue.empty():
            try:
                emissions.append(self._emissions_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        utterance_dur = self._utterance_samples / float(self.sample_rate)
        final_text = self._running_text
        result = StreamingResult(
            id=self._utterance_id,
            text=final_text,
            is_final=True,
            start=self._utterance_start_offset,
            duration=utterance_dur,
        )

        # Reset for next utterance
        self._absolute_offset = self._utterance_start_offset + utterance_dur
        self._stream_active = False
        self._utterance_samples = 0
        self._running_text = ""
        self._utterance_id = new_utterance_id()
        self._utterance_start_offset = self._absolute_offset

        non_final = [e for e in emissions if not e.is_final]
        if final_text:
            return non_final + [result]
        return non_final

    async def close(self) -> None:
        if self._stream_active:
            await self.end_utterance()

    # ── Internals ───────────────────────────────────────────────────────

    async def _start_stream(self) -> None:
        from google.genai import types

        prompt = "Transcribe the spoken audio verbatim. Output only the transcript text."
        if self.language:
            prompt = f"Transcribe the spoken audio verbatim in {self.language}. Output only the transcript text."

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=types.Content(
                role="user", parts=[types.Part(text=prompt)]
            ),
        )

        self._utterance_id = new_utterance_id()
        self._utterance_start_offset = self._absolute_offset
        self._utterance_samples = 0
        self._running_text = ""
        self._emissions_queue = asyncio.Queue()

        # The genai live API uses async context managers — we hold the
        # context object so we can close it on end_utterance.
        self._session_ctx = self._client.aio.live.connect(
            model=self.model, config=config,
        )
        self._session = await self._session_ctx.__aenter__()
        self._stream_active = True

        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Drain Gemini's response stream — push interim/final text events
        onto the emissions queue."""
        try:
            async for response in self._session.receive():
                # Each response can contain text, audio, or turn-complete
                if hasattr(response, "text") and response.text:
                    self._running_text = (self._running_text + response.text).strip()
                    em = StreamingResult(
                        id=self._utterance_id,
                        text=self._running_text,
                        is_final=False,
                        start=self._utterance_start_offset,
                        duration=self._utterance_samples / float(self.sample_rate),
                    )
                    self._emissions_queue.put_nowait(em)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # Error codes 1000 (OK) and 1006 (abnormal close) mean the server
            # ended the session. Treat as a graceful stream end (not a crash).
            err_str = str(exc)
            if "1000" in err_str or "1006" in err_str:
                log.debug("GeminiLiveStream %s session closed by server (%s)",
                          self._utterance_id, err_str.split(".")[0])
            else:
                log.exception("GeminiLiveStream %s receive loop error",
                              self._utterance_id)
        finally:
            # Always reset stream state so the next feed() opens a new session
            self._stream_active = False
            self._session = None
            self._session_ctx = None
