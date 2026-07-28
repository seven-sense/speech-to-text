"""
GoogleSTTStream — true streaming via Google Cloud Speech-to-Text v2's
`streaming_recognize` bidirectional gRPC.

Architecture:
- Establish a streaming session per utterance: send a `StreamingRecognizeRequest`
  with the config, then push `StreamingRecognizeRequest`s carrying audio
  chunks. Server returns a stream of `StreamingRecognizeResponse`s with
  `is_final=False` interims and `is_final=True` finals.
- We maintain a per-stream gRPC session: open it on construction, close + reopen
  on `end_utterance()` (the v2 API doesn't expose multi-utterance sessions in a
  way that maps cleanly to our boundaries).
- Latency: ~300ms first-word; transcription happens on Google's servers as
  audio arrives.

Auth: same as `GoogleSTTBackend` — Application Default Credentials via
GOOGLE_APPLICATION_CREDENTIALS, with GOOGLE_CLOUD_PROJECT for billing.

Cost: each audio second is billed; running this on multi-hour idle sessions
will accumulate charges. The session loop only sends audio when the VAD
endpointer says we're in a speaking phase, but Google still bills us for
the bytes we send.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Any, Iterator

import numpy as np

from ..asr_stream import StreamingResult, new_utterance_id, now_ms
from ..logger import get_logger

log = get_logger(__name__)

_STOP_SENTINEL = object()


@dataclass
class GoogleSTTStream:
    """Streaming wrapper over Google Cloud STT v2 streaming_recognize."""

    project_id: str
    location: str = "us-central1"
    model: str = "chirp_2"
    sample_rate: int = 16000
    language_codes: list[str] = field(default_factory=lambda: ["en-US"])
    max_utterance_sec: float = 30.0

    name: str = "google-stt-streaming"
    model_name: str = "chirp_2"

    # gRPC plumbing
    _client: Any = field(default=None, init=False, repr=False)
    _request_queue: Queue = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _response_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _emissions_queue: asyncio.Queue = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _utterance_id: str = field(default="", init=False)
    _utterance_start_offset: float = field(default=0.0, init=False)
    _absolute_offset: float = field(default=0.0, init=False)
    _utterance_samples: int = field(default=0, init=False)
    _last_interim_text: str = field(default="", init=False)
    _stream_active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        try:
            from google.cloud import speech_v2  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GoogleSTTStream requires `google-cloud-speech`. "
                "Install with: pip install -U google-cloud-speech"
            ) from e

        from google.cloud import speech_v2
        from google.api_core.client_options import ClientOptions

        self.model_name = self.model
        self.name = "google-stt-streaming"

        client_opts = None
        if self.location and self.location.lower() != "global":
            client_opts = ClientOptions(
                api_endpoint=f"{self.location}-speech.googleapis.com"
            )
        self._client = speech_v2.SpeechClient(client_options=client_opts)

        log.info("GoogleSTTStream ready (project=%s, location=%s, model=%s)",
                 self.project_id, self.location, self.model)
        print(f"[OK] Using Google Cloud STT (streaming, model: {self.model})")

    # ── Protocol ────────────────────────────────────────────────────────

    async def feed(self, pcm: np.ndarray) -> list[StreamingResult]:
        """Push PCM into the streaming session. Returns any emissions
        accumulated by the response-reader thread since the last call."""
        if pcm.size == 0:
            return []
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if pcm.ndim > 1:
            pcm = pcm.flatten()

        # Lazily start the gRPC stream on first frame of an utterance
        if not self._stream_active:
            self._start_stream()

        self._utterance_samples += pcm.size

        # Convert float32 → int16 LINEAR16 bytes
        int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
        self._request_queue.put(int16.tobytes())

        # Drain any responses that arrived since last call
        emissions: list[StreamingResult] = []
        while not self._emissions_queue.empty():
            try:
                emissions.append(self._emissions_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Force-flush if utterance is too long
        utterance_dur = self._utterance_samples / float(self.sample_rate)
        if utterance_dur >= self.max_utterance_sec:
            log.info("GoogleSTTStream %s force-flushing at %.1fs",
                     self._utterance_id, utterance_dur)
            emissions.extend(await self.end_utterance())
        return emissions

    async def end_utterance(self) -> list[StreamingResult]:
        """Close the current gRPC stream, flush remaining responses,
        return them as a single final."""
        if not self._stream_active:
            return []

        self._request_queue.put(_STOP_SENTINEL)

        # Wait for the response thread to finish (drains remaining responses)
        if self._response_thread is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, self._response_thread.join, 10.0
            )

        # Collect any remaining emissions
        emissions: list[StreamingResult] = []
        while not self._emissions_queue.empty():
            try:
                emissions.append(self._emissions_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        utterance_dur = self._utterance_samples / float(self.sample_rate)

        # Find the last interim or compose final from accumulated text
        final_text = ""
        for em in emissions:
            if em.is_final:
                final_text = em.text
        if not final_text and self._last_interim_text:
            final_text = self._last_interim_text

        result = StreamingResult(
            id=self._utterance_id,
            text=final_text,
            is_final=True,
            start=self._utterance_start_offset,
            duration=utterance_dur,
        )

        # Reset state for next utterance
        self._absolute_offset = self._utterance_start_offset + utterance_dur
        self._stream_active = False
        self._response_thread = None
        self._utterance_samples = 0
        self._utterance_id = new_utterance_id()
        self._utterance_start_offset = self._absolute_offset
        self._last_interim_text = ""

        # Return only interim emissions + the synthesized final
        non_final_emissions = [e for e in emissions if not e.is_final]
        if final_text:
            return non_final_emissions + [result]
        return non_final_emissions

    async def close(self) -> None:
        if self._stream_active:
            await self.end_utterance()

    # ── Internals ───────────────────────────────────────────────────────

    def _start_stream(self) -> None:
        from google.cloud.speech_v2.types import cloud_speech

        recognizer = (
            f"projects/{self.project_id}/locations/{self.location}/recognizers/_"
        )
        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=1,
            ),
            language_codes=self.language_codes,
            model=self.model,
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )
        streaming_features = cloud_speech.StreamingRecognitionFeatures(
            interim_results=True,
        )
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=streaming_features,
        )
        config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=streaming_config,
        )

        self._utterance_id = new_utterance_id()
        self._utterance_start_offset = self._absolute_offset
        self._utterance_samples = 0
        self._request_queue = Queue(maxsize=512)
        self._emissions_queue = asyncio.Queue()
        self._stream_active = True

        loop = asyncio.get_running_loop()
        self._response_thread = threading.Thread(
            target=self._run_stream,
            args=(config_request, loop),
            daemon=True,
            name=f"google-stt-stream-{self._utterance_id}",
        )
        self._response_thread.start()

    def _run_stream(self, config_request, loop: asyncio.AbstractEventLoop) -> None:
        """Worker thread: spin a request iterator, send to client, read responses,
        push StreamingResults onto the asyncio queue."""
        from google.cloud.speech_v2.types import cloud_speech

        def request_iter() -> Iterator:
            yield config_request
            while True:
                try:
                    item = self._request_queue.get(timeout=30.0)
                except Empty:
                    break
                if item is _STOP_SENTINEL:
                    break
                yield cloud_speech.StreamingRecognizeRequest(audio=item)

        try:
            responses = self._client.streaming_recognize(requests=request_iter())
            for response in responses:
                for result in response.results:
                    if not result.alternatives:
                        continue
                    text = result.alternatives[0].transcript.strip()
                    if not text:
                        continue
                    is_final = bool(getattr(result, "is_final", False))
                    em = StreamingResult(
                        id=self._utterance_id,
                        text=text,
                        is_final=is_final,
                        start=self._utterance_start_offset,
                        duration=self._utterance_samples / float(self.sample_rate),
                    )
                    if not is_final:
                        self._last_interim_text = text
                    # Hand off to the asyncio loop in a thread-safe way
                    loop.call_soon_threadsafe(self._emissions_queue.put_nowait, em)
        except Exception:
            log.exception("GoogleSTTStream %s response thread crashed",
                          self._utterance_id)
