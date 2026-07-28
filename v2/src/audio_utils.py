"""
Audio processing utilities.
"""

import numpy as np
import torch
from pydub import AudioSegment
from collections import deque
import threading


def segment_to_float32(seg: AudioSegment) -> np.ndarray:
    """
    Convert AudioSegment to float32 numpy array.

    Args:
        seg: AudioSegment to convert

    Returns:
        Float32 numpy array normalized to [-1, 1]
    """
    samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def load_audio(audio_path, sample_rate=16000):
    """
    Load audio file and convert to standard format.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate

    Returns:
        AudioSegment in mono, 16-bit, specified sample rate
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
    return audio


def prepare_waveform(audio):
    """
    Prepare waveform tensor for pyannote pipeline.

    Args:
        audio: AudioSegment

    Returns:
        Torch tensor of shape (1, samples)
    """
    wave = segment_to_float32(audio)
    return torch.from_numpy(wave).unsqueeze(0)


def sanitize_for_filename(name: str) -> str:
    """Make a string (e.g. a model id) safe for use in a filename."""
    # Strip HF org prefix (e.g. "Qwen/Qwen3-ASR-0.6B" -> "Qwen3-ASR-0.6B")
    name = name.rsplit("/", 1)[-1]
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def detect_silence(audio_data, threshold=0.01):
    """
    Simple silence detection based on RMS.

    Args:
        audio_data: Numpy array of audio samples
        threshold: RMS threshold for silence
 
    Returns:
        True if silence detected
    """
    if len(audio_data) == 0:
        return True
    rms = np.sqrt(np.mean(audio_data ** 2))
    return rms < threshold


class AudioBuffer:
    """Thread-safe circular buffer for real-time audio data."""

    def __init__(self, max_duration_seconds, sample_rate):
        """
        Initialize audio buffer.

        Args:
            max_duration_seconds: Maximum buffer duration
            sample_rate: Audio sample rate
        """
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()

    def add(self, audio_data):
        """Add audio data to buffer."""
        with self.lock:
            self.buffer.extend(audio_data)

    def get_audio(self, duration_seconds=None):
        """
        Get audio data from buffer (last-N semantics — back-compat).

        Args:
            duration_seconds: Duration to retrieve (None for all)

        Returns:
            Numpy array of audio samples
        """
        with self.lock:
            if duration_seconds is None:
                return np.array(self.buffer, dtype=np.float32)

            num_samples = int(duration_seconds * self.sample_rate)
            num_samples = min(num_samples, len(self.buffer))

            if num_samples == 0:
                return np.array([], dtype=np.float32)

            # Get last N samples
            return np.array(list(self.buffer)[-num_samples:], dtype=np.float32)

    # ── Front-anchored access (Task 5) ────────────────────────────────────

    def peek_front(self, duration_seconds=None) -> np.ndarray:
        """
        Non-destructive read of the *oldest* unprocessed audio (front of FIFO).

        Unlike `get_audio`, this returns samples from the **front** of the
        buffer, not the tail. Combined with `consume()`, this gives FIFO
        semantics needed for streaming: process the oldest pending audio,
        advance the front pointer, leave any tail (including audio that
        arrived during transcription) intact for the next pass.
        """
        with self.lock:
            if duration_seconds is None:
                return np.array(self.buffer, dtype=np.float32)
            num_samples = int(duration_seconds * self.sample_rate)
            num_samples = min(num_samples, len(self.buffer))
            if num_samples == 0:
                return np.array([], dtype=np.float32)
            # Slicing a deque is O(N) but acceptable; for this workload N is
            # at most buffer_duration * sample_rate (e.g., 15s * 16kHz = 240k)
            # and we run this once per chunk_duration tick (~2s).
            return np.array(list(self.buffer)[:num_samples], dtype=np.float32)

    def consume(self, num_samples: int) -> int:
        """
        Drop `num_samples` from the *front* of the buffer. Returns the actual
        number consumed (might be less if the buffer doesn't have that many).
        """
        if num_samples <= 0:
            return 0
        with self.lock:
            num_samples = min(num_samples, len(self.buffer))
            for _ in range(num_samples):
                self.buffer.popleft()
            return num_samples

    def consume_seconds(self, seconds: float) -> int:
        return self.consume(int(seconds * self.sample_rate))

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()

    def __len__(self):
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)
