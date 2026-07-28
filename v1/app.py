import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

import os
import gc
import time
import numpy as np
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper

# Import our custom separate logging system
from logger import logger, log_section, instrument_function

AUDIO_FILE = "audio.mp3"
OUT_FILE = "transcript.txt"

# ---------------- TUNING ----------------
MIN_SEG_DUR = 0.60    # seconds: drop tiny diarization segments
MERGE_GAP   = 0.35    # seconds: merge same-speaker if gap <= this
# ----------------------------------------

logger.info("Speech-to-Text Pipeline execution started.")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.critical("HF_TOKEN environment variable is required.")
    raise ValueError(
        "HF_TOKEN environment variable is required. "
        "Please set it using: export HF_TOKEN=your_token_here"
    )

with log_section("Loading Hugging Face Pyannote Diarization Pipeline"):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

with log_section("Loading and Preprocessing Audio file"):
    audio = AudioSegment.from_file(AUDIO_FILE)
    logger.info(f"Original Audio details: duration={audio.duration_seconds:.2f}s, channels={audio.channels}, sample_rate={audio.frame_rate}Hz")
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    logger.info("Audio normalized to mono, 16kHz, 16-bit depth.")

with log_section("Loading OpenAI Whisper Model"):
    model = whisper.load_model("small.en")  # try "medium.en" for better quality


@instrument_function
def segment_to_float32(seg: AudioSegment) -> np.ndarray:
    """
    Converts pydub AudioSegment samples to normalized float32 numpy array.
    Time Complexity: O(N) where N is number of audio samples.
    Space Complexity: O(N) to hold the resulting float32 array in memory.
    """
    logger.debug(f"Converting segment to float32: duration={seg.duration_seconds:.2f}s, channels={seg.channels}, rate={seg.frame_rate}Hz")
    samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


@instrument_function
def merge_same_speaker(segments, merge_gap=0.35):
    """
    Merge consecutive segments if they have same speaker and the gap is small.
    segments: list of (start, end, speaker) sorted by start.
    
    Time Complexity: O(N) where N is the number of segments. We loop through the list once.
    Space Complexity: O(N) to store the merged segments list.
    """
    if not segments:
        logger.debug("merge_same_speaker: received empty segments list.")
        return []
    
    logger.info(f"Merging same-speaker segments. Input count: {len(segments)}, Gap threshold: {merge_gap}s")
    merged = [[segments[0][0], segments[0][1], segments[0][2]]]
    merge_count = 0
    for s, e, spk in segments[1:]:
        ps, pe, pspk = merged[-1]
        if spk == pspk and (s - pe) <= merge_gap:
            merged[-1][1] = max(pe, e)
            merge_count += 1
        else:
            merged.append([s, e, spk])
            
    logger.info(f"Merging complete. Output count: {len(merged)} (merged {merge_count} segments) | Time Complexity: O(n), Space Complexity: O(n)")
    return [(s, e, spk) for s, e, spk in merged]


@instrument_function
def split_overlaps(segments):
    """
    Convert possibly-overlapping segments into a clean non-overlapping timeline
    by splitting earlier segments around later segments.

    Input: list of (start, end, speaker) sorted by start.
    Output: list of (start, end, speaker) sorted, non-overlapping.
    
    Time Complexity: O(N) where N is the number of segments. Loops through once.
    Space Complexity: O(N) to hold the split list.
    """
    if not segments:
        logger.debug("split_overlaps: received empty segments list.")
        return []

    logger.info(f"Resolving overlapping speaker segments. Input count: {len(segments)}")
    out = []
    splits_performed = 0
    for s, e, spk in segments:
        if not out:
            out.append([s, e, spk])
            continue

        ps, pe, pspk = out[-1]

        # No overlap
        if s >= pe:
            out.append([s, e, spk])
            continue

        # Overlap exists: s < pe
        splits_performed += 1
        logger.debug(f"Overlap detected: Prev [{ps:.2f}s -- {pe:.2f}s] {pspk} vs Next [{s:.2f}s -- {e:.2f}s] {spk}")
        if s > ps:
            out[-1][1] = s
        else:
            out.pop()

        out.append([s, e, spk])

        if pe > e:
            out.append([e, pe, pspk])

    final_out = [seg for seg in out if seg[1] > seg[0]]
    logger.info(f"Overlap resolution complete. Output count: {len(final_out)} (performed {splits_performed} splits/truncations) | Time Complexity: O(n), Space Complexity: O(n)")
    return [(s, e, spk) for s, e, spk in final_out]


@instrument_function
def build_clean_segments(diarization_annotation):
    """
    Filters out tiny segments, merges same speaker turns, splits overlapping turns.
    Time Complexity: O(N log N) dominated by sorting the raw segments, then O(N) for linear processing.
    Space Complexity: O(N) to hold intermediate lists.
    """
    segs = []
    raw_count = 0
    discarded_count = 0
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        raw_count += 1
        s = float(turn.start)
        e = float(turn.end)
        if (e - s) < MIN_SEG_DUR:
            discarded_count += 1
            continue
        segs.append((s, e, speaker))

    logger.info(f"Timeline extraction: raw={raw_count}, discarded noise (<{MIN_SEG_DUR}s)={discarded_count}, kept={len(segs)}")

    # Sort
    segs.sort(key=lambda x: x[0])

    # Merge same speaker (reduces fragmentation)
    segs = merge_same_speaker(segs, merge_gap=MERGE_GAP)

    # Split overlaps to create a clean timeline (no nested timestamps)
    segs = split_overlaps(segs)

    # Merge again (splitting can create adjacent same-speaker pieces)
    segs = merge_same_speaker(segs, merge_gap=0.01)

    return segs


# ---- Provide waveform directly to pyannote (bypasses torchcodec decoding) ----
with log_section("Converting Full Audio to Float32 Waveform"):
    full_wave = segment_to_float32(audio)
    full_waveform = torch.from_numpy(full_wave).unsqueeze(0)  # (1, time)
    logger.info(f"Full waveform tensor shape={full_waveform.shape}, dtype={full_waveform.dtype}")

with log_section("Running Pyannote Speaker Diarization"):
    output = pipeline({"waveform": full_waveform, "sample_rate": 16000})
    diar = output.speaker_diarization  # pyannote 4.x

with log_section("Optimizing Speaker Segments Timeline"):
    segments = build_clean_segments(diar)

# Get unique speakers and log info
speakers = sorted(list(set(seg[2] for seg in segments)))
total_spks = len(speakers)
logger.info(
    f"Speaker Diarization complete: Found {total_spks} unique speakers: {speakers}",
    extra={"total_speakers": total_spks}
)

total_segments = len(segments)
logger.info(
    f"Starting Whisper transcription for {total_segments} segments.",
    extra={"total_speakers": total_spks}
)

with log_section(f"Transcribing {total_segments} audio segments"):
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for idx, (s, e, speaker) in enumerate(segments, 1):
            logger.info(
                f"Transcribing segment {idx}/{total_segments}: [{s:.2f}s -- {e:.2f}s] for {speaker}",
                extra={
                    "speaker": speaker,
                    "segment_start": f"{s:.2f}",
                    "segment_end": f"{e:.2f}",
                    "total_speakers": total_spks
                }
            )
            seg_start_time = time.perf_counter()

            seg_audio = audio[int(s * 1000):int(e * 1000)]
            waveform = segment_to_float32(seg_audio)

            # Whisper decoding settings for more stability
            result = model.transcribe(
                waveform,
                fp16=False,
                temperature=0.0,
                beam_size=5,
            )

            text = result.get("text", "").strip()
            detected_lang = result.get("language", "en")
            seg_duration = time.perf_counter() - seg_start_time
            if text:
                logger.info(
                    f"Segment {idx} transcribed successfully in {seg_duration:.2f}s: \"{text[:50]}...\"",
                    extra={
                        "speaker": speaker,
                        "language": detected_lang,
                        "segment_start": f"{s:.2f}",
                        "segment_end": f"{e:.2f}",
                        "total_speakers": total_spks
                    }
                )
                f.write(f"\n[{s:.2f}s -- {e:.2f}s] {speaker}: {text}")
            else:
                logger.warning(
                    f"Segment {idx} returned empty transcription.",
                    extra={
                        "speaker": speaker,
                        "language": detected_lang,
                        "segment_start": f"{s:.2f}",
                        "segment_end": f"{e:.2f}",
                        "total_speakers": total_spks
                    }
                )

            del seg_audio, waveform, result
            gc.collect()

logger.info(f"Done. Wrote transcript to: {OUT_FILE}")
