"""
Real-time transcription from microphone input using NVIDIA NeMo.
"""

import sys
import time
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd

from .transcriber import BaseTranscriber, is_likely_hallucination
from .audio_utils import AudioBuffer, detect_silence, sanitize_for_filename
from .audio_recorder import RollingMP3Writer, recording_base_path
from .chunking import BoundaryStrategy, find_endpoint, dedupe_overlap
from .config import get_config
from .logger import get_logger

log = get_logger(__name__)


class RealtimeTranscriber(BaseTranscriber):
    """Real-time multi-speaker transcription from microphone using NeMo."""

    def __init__(self, nemo_model=None, config=None):
        """
        Initialize real-time transcriber.

        Args:
            nemo_model: NeMo model name (overrides config)
            config: Config instance
        """
        self.config = config or get_config()

        # Initialize with realtime model
        super().__init__(nemo_model, self.config, use_realtime_model=True)

        # Initialize audio buffer
        self.audio_buffer = AudioBuffer(
            self.config.buffer_duration,
            self.config.sample_rate
        )

        # State management
        self.is_running = False
        self.processing_offset = 0.0
        self.transcript_log = []
        self._last_emitted_text = ""  # for overlap-mode dedup

        # Setup output file if needed.
        # Filename = realtime_<DD-MM-YY_HH-MM-SS>.txt — concise. Backend +
        # model are recorded in the file's header instead of the filename.
        if self.config.save_transcript:
            timestamp = datetime.now().strftime(self.config.timestamp_format)
            path = self.config.transcript_dir / f"realtime_{timestamp}.txt"
            counter = 1
            while path.exists():
                counter += 1
                path = self.config.transcript_dir / f"realtime_{timestamp}_{counter}.txt"
            self.transcript_path = path
            self.transcript_file = open(self.transcript_path, "w", encoding="utf-8")
            self.transcript_file.write(f"Real-time Transcript ({self.asr_backend.name.upper()} ASR)\n")
            self.transcript_file.write(f"Model: {self.asr_backend.model_name}\n")
            self.transcript_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.transcript_file.write("=" * 60 + "\n\n")

        # Optional audio recording (Task 4)
        self.recorder = None
        if self.config.recording_enabled:
            base = recording_base_path(
                prefix="realtime",
                asr_backend_name=self.asr_backend.name,
                asr_model_name=self.asr_backend.model_name,
                recordings_dir=self.config.recording_dir,
            )
            self.recorder = RollingMP3Writer(
                base_path=base,
                sample_rate=self.config.sample_rate,
                max_bytes=self.config.recording_max_part_bytes,
                bitrate_kbps=self.config.recording_bitrate_kbps,
                fmt=self.config.recording_format,
            )
            log.info("Recording enabled: %s_NNN.%s",
                     base.name, self.config.recording_format)

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - called by sounddevice."""
        if status:
            log.warning("sounddevice status: %s", status)

        # Convert to mono and normalize
        audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
        audio_data = audio_data.flatten().astype(np.float32)

        # Add to buffer
        self.audio_buffer.add(audio_data)

        # Tee to recorder (non-blocking; encode happens in a worker thread)
        if self.recorder is not None:
            self.recorder.write(audio_data)

    def process_audio_chunk(self):
        """Process accumulated audio chunk using the configured boundary strategy.

        Front-anchored read + consume gives proper FIFO semantics: audio that
        arrives during transcription stays in the buffer for the next tick
        instead of being silently overwritten by the "last-N samples" view.
        """
        sr = self.config.sample_rate
        strategy = self._boundary_strategy()

        # ── boundary_strategy="none" — back-compat path (legacy behavior) ──
        if strategy == BoundaryStrategy.NONE:
            audio_data = self.audio_buffer.get_audio(self.config.chunk_duration)
            if len(audio_data) == 0:
                return
            if detect_silence(audio_data, self.config.silence_threshold):
                self.processing_offset += self.config.chunk_duration
                return
            print(f"\r[Processing audio...]", end='', flush=True)
            self._run_diarize_and_emit(audio_data, base_offset=self.processing_offset)
            self.processing_offset += self.config.chunk_duration
            return

        # ── New front-anchored path (vad / overlap) ──
        # Peek up to max_chunk_duration of the OLDEST pending audio
        max_dur = self.config.max_chunk_duration
        pcm = self.audio_buffer.peek_front(max_dur)
        if len(pcm) == 0:
            return

        chunk_dur = self.config.chunk_duration
        min_chunk_samples = int(chunk_dur * sr)
        # Need at least ~chunk_duration before we'd consider processing,
        # otherwise we wait for more audio (the loop ticks again in 0.1s).
        if len(pcm) < min_chunk_samples:
            return

        # Silence skip — but consume so the silent audio doesn't sit forever
        if detect_silence(pcm, self.config.silence_threshold):
            consumed = self.audio_buffer.consume(int(chunk_dur * sr))
            self.processing_offset += consumed / sr
            return

        print(f"\r[Processing audio...]", end='', flush=True)

        # Pick the cut point
        try:
            cut_samples, overlap_keepalive = self._pick_cut(
                pcm, sr, strategy
            )
        except Exception:
            log.exception("Cut-point selection failed; force-cutting at chunk_duration")
            cut_samples = min_chunk_samples
            overlap_keepalive = 0

        if cut_samples is None:
            # Not yet enough audio for a clean cut — wait for the next tick.
            return

        cut_pcm = pcm[:cut_samples]
        try:
            self._run_diarize_and_emit(
                cut_pcm,
                base_offset=self.processing_offset,
                strategy=strategy,
            )
        except Exception as e:
            log.exception("process_audio_chunk failed: %s", e)
            if "CUDA" not in str(e):
                print(f"\r[Error: {e}]", file=sys.stderr)

        # Advance the buffer + offset by everything we processed except the
        # overlap-keepalive tail (which we want to re-read next tick).
        advance_samples = max(0, cut_samples - overlap_keepalive)
        if advance_samples > 0:
            consumed = self.audio_buffer.consume(advance_samples)
            self.processing_offset += consumed / sr

    def _boundary_strategy(self) -> BoundaryStrategy:
        try:
            return BoundaryStrategy(self.config.boundary_strategy)
        except (ValueError, AttributeError):
            return BoundaryStrategy.VAD

    def _pick_cut(self, pcm: np.ndarray, sr: int,
                  strategy: BoundaryStrategy) -> tuple[int | None, int]:
        """Returns (cut_sample_index_or_None, overlap_keepalive_samples)."""
        chunk_samples = int(self.config.chunk_duration * sr)
        max_samples = int(self.config.max_chunk_duration * sr)

        if strategy == BoundaryStrategy.OVERLAP:
            cut = min(chunk_samples, len(pcm))
            overlap = int(self.config.overlap_duration * sr)
            return cut, overlap

        # VAD strategy
        vad_segs = self.detect_voice_activity(pcm)
        cut = find_endpoint(
            len(pcm), sr, vad_segs,
            min_chunk_s=max(self.config.min_segment_duration, 1.0),
            max_chunk_s=self.config.max_chunk_duration,
            min_silence_ms=self.config.min_silence_ms,
        )
        if cut is None:
            # No clean cut found yet. If buffer has reached max, force-cut at max.
            if len(pcm) >= max_samples:
                return max_samples, 0
            # Otherwise, wait for more audio.
            return None, 0
        return cut, 0

    def _run_diarize_and_emit(self, audio_data: np.ndarray,
                              base_offset: float,
                              strategy: BoundaryStrategy = BoundaryStrategy.NONE):
        segments = self.perform_diarization(audio_data)
        if not segments:
            return  # VAD found no speech — don't transcribe pure noise

        for start, end, speaker in segments:
            start_sample = int(start * self.config.sample_rate)
            end_sample = int(end * self.config.sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            text = self.transcribe_segment(segment_audio)
            seg_dur = end - start
            if not text:
                continue
            if is_likely_hallucination(text, seg_dur):
                log.debug("Dropping likely hallucination (%.2fs → %r)",
                          seg_dur, text)
                continue
            if strategy == BoundaryStrategy.OVERLAP and self._last_emitted_text:
                text = dedupe_overlap(self._last_emitted_text, text)
                if not text:
                    continue
            abs_start = base_offset + start
            abs_end = base_offset + end
            self._output_transcript(abs_start, abs_end, speaker, text)
            self._last_emitted_text = text

    def _output_transcript(self, start, end, speaker, text):
        """Output transcript line."""
        line = self.format_transcript_line(start, end, speaker, text)

        if self.config.print_to_console:
            # Clear line and print cleanly
            print(f"\r{' ' * 100}\r{line}", flush=True)

        self.transcript_log.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text
        })

        if self.config.save_transcript:
            self.transcript_file.write(line + "\n")
            self.transcript_file.flush()

    def processing_loop(self):
        """Main processing loop - runs in separate thread."""
        last_process_time = time.time()

        while self.is_running:
            current_time = time.time()

            # Process every chunk_duration seconds
            if current_time - last_process_time >= self.config.chunk_duration:
                self.process_audio_chunk()
                last_process_time = current_time

            time.sleep(0.1)

    def start(self):
        """Start real-time transcription."""
        backend_label = {"nemo": "NVIDIA NeMo", "qwen": "Qwen3-ASR"}.get(
            self.asr_backend.name, self.asr_backend.name
        )
        print("\n" + "=" * 80)
        print(f"  REAL-TIME MULTI-SPEAKER TRANSCRIPTION ({backend_label})")
        print("=" * 80)
        if self.config.save_transcript:
            print(f"  📝 Saving to: {self.transcript_path.name}")
        print(f"  🎤 Model: {self.asr_backend.model_name}")
        print("\n  Press Ctrl+C to stop\n")
        print("=" * 80 + "\n")

        self.is_running = True
        log.info("Realtime session start — backend=%s model=%s",
                 self.asr_backend.name, self.asr_backend.model_name)

        # Start processing thread
        processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        processing_thread.start()

        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                callback=self.audio_callback,
                dtype=np.float32
            ):
                print("  🔴 LIVE - Speak into your microphone...\n")

                while self.is_running:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            log.info("Realtime transcription stopped by user (Ctrl+C)")
            print("\n\nStopping transcription...")

        except Exception as e:
            log.exception("Realtime transcription crashed: %s", e)
            print(f"\nError: {e}", file=sys.stderr)

        finally:
            self.stop()

    def stop(self):
        """Stop transcription and cleanup."""
        self.is_running = False

        # Process remaining audio
        print("\nProcessing remaining audio...")
        self.process_audio_chunk()

        # Flush + close audio recorder
        if self.recorder is not None:
            try:
                last_part = self.recorder.close()
                parts = self.recorder.parts
                if parts:
                    log.info("Recording saved: %d part(s), last: %s",
                             len(parts), last_part)
                    print(f"[OK] Recording saved: {len(parts)} part(s) "
                          f"in {self.config.recording_dir}")
            except Exception:
                log.exception("Recorder close failed")

        # Close file
        if self.config.save_transcript and hasattr(self, 'transcript_file'):
            self.transcript_file.write(f"\n{'=' * 60}\n")
            self.transcript_file.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.transcript_file.close()
            log.info("Transcript saved: %s", self.transcript_path)
            print(f"\n[OK] Transcript saved to: {self.transcript_path}")

        log.info("Realtime session end — total segments=%d", len(self.transcript_log))
        print(f"[OK] Total segments: {len(self.transcript_log)}")
        print("\nGoodbye!")
