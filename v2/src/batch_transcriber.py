"""
Batch transcription for audio files (no diarization).
"""

import gc
from pathlib import Path
from datetime import datetime

from .transcriber import BaseTranscriber, is_likely_hallucination
from .audio_utils import load_audio, segment_to_float32
from .audio_recorder import chunk_audio_file
from .config import get_config
from .logger import get_logger

log = get_logger(__name__)


class BatchTranscriber(BaseTranscriber):
    """Batch audio file transcription. Transcript-only — no speaker labels."""

    def __init__(self, nemo_model=None, config=None):
        super().__init__(nemo_model, config, use_realtime_model=False)

    def transcribe_file(self, audio_path, output_path=None, handle_overlaps=None):
        """
        Transcribe an audio file to a single transcript file.

        For long inputs (>batch_window_sec), the file is streamed through
        fixed-duration windows so memory stays bounded.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if output_path is None:
            timestamp = datetime.now().strftime(self.config.timestamp_format)
            output_path = self.config.transcript_dir / f"{audio_path.stem}_{timestamp}.txt"
        else:
            output_path = Path(output_path)

        log.info("Batch session start — input=%s output=%s backend=%s model=%s",
                 audio_path, output_path, self.asr_backend.name,
                 self.asr_backend.model_name)
        print(f"Processing: {audio_path}")
        print(f"Output: {output_path}\n")

        # Probe duration without loading the whole file
        from pydub.utils import mediainfo
        try:
            duration_s = float(mediainfo(str(audio_path)).get("duration", 0))
        except Exception:
            duration_s = 0.0

        window_s = self.config.batch_window_sec
        if duration_s > window_s:
            print(f"Long input ({duration_s:.0f}s) — streaming in "
                  f"{window_s:.0f}s windows...")
            transcripts = self._transcribe_windowed(audio_path, window_s)
        else:
            print("Loading audio...")
            audio = load_audio(audio_path, self.config.sample_rate)
            audio_data = segment_to_float32(audio)
            transcripts = self._transcribe_segment_or_window(
                audio_data, offset=0.0,
            )

        self._write_transcript(output_path, audio_path, transcripts)

        log.info("Batch session end — segments=%d output=%s",
                 len(transcripts), output_path)
        print(f"\n[OK] Done! Transcribed {len(transcripts)} segment(s) "
              f"with {self.asr_backend.name.upper()}")
        print(f"[OK] Saved to: {output_path}")
        return transcripts

    def _transcribe_segment_or_window(self, audio_data, offset: float = 0.0) -> list[dict]:
        """Transcribe an audio array as one segment. No VAD."""
        duration = len(audio_data) / float(self.config.sample_rate)
        print(f"Transcribing {duration:.1f}s with {self.asr_backend.name.upper()} ASR...")
        text = self.transcribe_segment(audio_data)
        if not text:
            return []
        if is_likely_hallucination(text, duration):
            log.debug("Dropping likely hallucination (%.2fs → %r)",
                      duration, text)
            return []
        return [{"start": offset, "end": offset + duration, "text": text}]

    def _transcribe_windowed(self, audio_path: Path, window_s: float) -> list[dict]:
        """Stream a long file through fixed-duration windows."""
        all_transcripts: list[dict] = []
        for window_offset, pcm in chunk_audio_file(
            audio_path, window_s, self.config.sample_rate
        ):
            print(f"\n── Window {window_offset:.0f}s — "
                  f"{window_offset + len(pcm)/self.config.sample_rate:.0f}s ──")
            transcripts = self._transcribe_segment_or_window(
                pcm, offset=window_offset,
            )
            all_transcripts.extend(transcripts)
            del pcm
            gc.collect()
        return all_transcripts

    def _write_transcript(self, output_path, audio_path, transcripts):
        """Write transcript to file (no speaker labels)."""
        backend_label = {
            "nemo": "NVIDIA NeMo", "qwen": "Qwen3-ASR",
            "gemini": "Gemini", "google-stt": "Google Cloud STT",
            "whisper": "Whisper",
        }.get(self.asr_backend.name, self.asr_backend.name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Transcript for: {audio_path.name}\n")
            f.write(f"Generated with: {backend_label}\n")
            f.write(f"Model: {self.asr_backend.model_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            for item in transcripts:
                line = (f"[{item['start']:.2f}s - {item['end']:.2f}s] "
                        f"{item['text']}")
                f.write(line + "\n")
