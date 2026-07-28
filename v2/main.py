#!/usr/bin/env python3
"""
Speech-to-Text Multi-Speaker Transcription
Main entry point for real-time and batch transcription.
"""

import sys
import argparse
from pathlib import Path

from src.config import get_config
from src.logger import setup_logging, get_logger


def _init_logging():
    """Initialize logging from config (must run before importing transcribers
    so their module-level loggers attach to configured handlers)."""
    cfg = get_config()
    setup_logging(
        log_dir=cfg.log_dir,
        file_level=cfg.log_file_level,
        console_level=cfg.log_console_level,
        file_enabled=cfg.log_file_enabled,
        console_enabled=cfg.log_console_enabled,
    )


_init_logging()
log = get_logger(__name__)





def _backend_label(config):
    """Human-readable label for the active ASR backend."""
    backend = (config.asr_backend or "nemo").lower()
    return {
        "nemo": "NVIDIA NeMo",
        "qwen": "Qwen3-ASR",
        "gemini": "Gemini",
        "google-stt": "Google Cloud STT",
        "whisper": "OpenAI Whisper (faster-whisper)",
    }.get(backend, backend)


def realtime_mode(args):
    """Run real-time transcription mode."""
    from src.realtime_transcriber import RealtimeTranscriber
    config = get_config()
    print(f"Starting real-time transcription with {_backend_label(config)}...")

    transcriber = RealtimeTranscriber(nemo_model=args.model, config=config)
    transcriber.start()


def serve_mode(args):
    """Run the multi-user web server."""
    config = get_config()
    backend_label = _backend_label(config)
    print(f"Starting realtime web server with {backend_label}...")

    # Mic access requires a secure context. Only http://localhost,
    # http://127.0.0.1, and HTTPS qualify. http://0.0.0.0 and LAN IPs do NOT.
    # Always point users at the literal URL that works.
    print()
    print("=" * 70)
    print(f"  --> Open this URL in your browser:  http://localhost:{args.port}")
    print("=" * 70)
    print("  [WARN] Use 'localhost' literally. Don't paste 'http://0.0.0.0:...' from")
    print("    the uvicorn startup line below — browsers block mic access on")
    print("    non-secure origins (0.0.0.0, LAN IPs, etc.).")
    print("=" * 70)
    print()

    # Defer import so the FastAPI/uvicorn deps aren't required for non-server modes.
    from src.server import run as server_run
    server_run(host=args.host, port=args.port, reload=args.reload)


def batch_mode(args):
    """Run batch transcription mode."""
    from src.batch_transcriber import BatchTranscriber
    audio_file = Path(args.audio_file)

    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    config = get_config()
    print(f"Starting batch transcription with {_backend_label(config)} for: {audio_file}")

    transcriber = BatchTranscriber(nemo_model=args.model, config=config)

    transcriber.transcribe_file(
        audio_path=audio_file,
        output_path=args.output
    )


def benchmark_mode(args):
    """Run a benchmark dataset through the chosen ASR backend, write WER reports."""
    from src.asr_backends import make_asr_backend
    from src.benchmark import (
        Evaluator, load_dataset, write_json, write_markdown_gap_assessment
    )

    config = get_config()
    backend_name = (args.backend or config.asr_backend or "nemo").lower()
    is_cloud = backend_name in ("gemini", "google-stt")

    # Cost guard for cloud backends
    if is_cloud and (args.limit is None or args.limit > 20) and not args.confirm_paid:
        print(f"Refusing to run a paid backend ({backend_name}) over more than "
              f"20 samples without --confirm-paid. Re-run with "
              f"`--limit 20` for a free-tier-safe smoke test, or pass "
              f"`--confirm-paid` to acknowledge billing.")
        sys.exit(2)

    # Build backend (uses default model from config if not overridden)
    if args.model:
        model_name = args.model
    else:
        # Use the realtime default of the backend (smaller / cheaper); the
        # batch defaults are tuned for transcript quality, not bench speed.
        if backend_name == "nemo":
            model_name = config.nemo_realtime_model
        elif backend_name == "qwen":
            model_name = config.qwen_realtime_model
        elif backend_name == "gemini":
            model_name = config.gemini_realtime_model
        elif backend_name == "google-stt":
            model_name = config.google_stt_realtime_model
        elif backend_name == "whisper":
            model_name = config.whisper_realtime_model
        else:
            print(f"Unknown backend: {backend_name}")
            sys.exit(1)

    print(f"Building ASR backend: {backend_name} / {model_name}")
    backend = make_asr_backend(backend_name, model_name, config)

    print(f"Loading dataset: {args.dataset} (split={args.split})")
    loader = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=config.benchmark_cache_dir,
        allow_download=args.allow_download,
    )

    evaluator = Evaluator(
        backend=backend,
        cache_dir=config.benchmark_cache_dir,
        sample_rate=config.sample_rate,
        use_cache=not args.no_cache,
    )

    print(f"Running evaluator (limit={args.limit})...")
    results = evaluator.run(loader, limit=args.limit)

    # Decide output paths
    output_json = (Path(args.output) if args.output else
                   config.benchmark_output_dir /
                   f"{backend_name}_{model_name.replace('/', '_')}_{args.dataset}_{args.split}.json")
    output_md = output_json.with_suffix(".md")

    meta = {
        "backend": backend_name,
        "model": model_name,
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit if args.limit is not None else "all",
        "samples_run": len(results),
    }
    write_json(results, output_json, meta=meta)
    write_markdown_gap_assessment(results, output_md, meta=meta)

    print()
    print(f"[OK] JSON report:     {output_json}")
    print(f"[OK] Markdown report: {output_md}")
    if results:
        mean_wer = sum(r.wer for r in results) / len(results)
        print(f"[OK] Mean WER over {len(results)} samples: {mean_wer:.4f}")


def verify_setup():
    """Verify system setup."""
    print("Verifying setup...")
    print()

    try:
        # Check config
        config = get_config()
        print("[OK] Configuration loaded")
        if config.hf_token:
            print(f"  - HF Token: {'*' * min(len(config.hf_token), 20)}... (set)")
        else:
            print("  - HF Token: not set (optional — needed only for gated HF models)")
        print(f"  - Sample Rate: {config.sample_rate} Hz")
        print(f"  - ASR backend: {config.asr_backend}")
        if config.asr_backend.lower() == "qwen":
            print(f"  - Qwen ASR (realtime): {config.qwen_realtime_model}")
            print(f"  - Qwen ASR (batch): {config.qwen_batch_model}")
        else:
            print(f"  - NeMo ASR (realtime): {config.nemo_realtime_model}")
            print(f"  - NeMo ASR (batch): {config.nemo_batch_model}")
        print(f"  - NeMo VAD: {getattr(config, 'nemo_vad_model', 'N/A')}")
        print(f"  - NeMo Speaker: {getattr(config, 'nemo_speaker_model', 'N/A')}")
        print()

        # Check dependencies
        print("Checking dependencies...")
        import numpy
        print("  [OK] numpy")
        import torch
        print("  [OK] torch")
        import pydub
        print("  [OK] pydub")
        import nemo.collections.asr as nemo_asr
        print("  [OK] nemo_toolkit[asr]")
        from sklearn.cluster import SpectralClustering
        print("  [OK] scikit-learn")
        import sounddevice
        print("  [OK] sounddevice")
        import dotenv
        print("  [OK] python-dotenv")
        print()

        # Check PyTorch device
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("[OK] MPS (Apple Silicon) available")
        else:
            print("[OK] CPU only (consider GPU for faster processing)")
        print()

        # Check audio devices (non-fatal — containers/headless hosts have no mic)
        try:
            devices = sounddevice.query_devices()
            print(f"[OK] Found {len(devices)} audio device(s)")
            try:
                default_input = sounddevice.query_devices(kind='input')
                print(f"  - Default input: {default_input['name']}")
            except Exception:
                print("  - No default input device (realtime mic mode unavailable; "
                      "batch mode still works)")
        except Exception as e:
            print(f"[WARN] No audio devices accessible: {e}")
            print("  Realtime mic mode will not work here. Batch mode is unaffected.")
            print("  (Common in Docker on Mac/Windows, or on headless servers.)")
        print()

        # Check FFmpeg
        import subprocess
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("[OK] FFmpeg installed")
            else:
                print("[FAIL] FFmpeg not found (required for audio processing)")
        except FileNotFoundError:
            print("[FAIL] FFmpeg not found (install with: brew install ffmpeg)")
        print()

        print("=" * 60)
        print("[OK] Setup verification complete!")
        print("=" * 60)
        print()
        print("Quick start:")
        print("  Real-time: python main.py realtime")
        print("  Batch:     python main.py batch audio.mp3")
        print()

        return 0

    except Exception as e:
        print(f"\n[FAIL] Setup verification failed: {e}")
        print("\nPlease ensure:")
        print("  1. All dependencies installed: pip install -r requirements.txt")
        print("  2. FFmpeg installed: brew install ffmpeg (Mac) / apt install ffmpeg (Linux)")
        print("  3. HF_TOKEN set in .env (only needed for gated HuggingFace models)")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text Multi-Speaker Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time transcription
  python main.py realtime

  # Real-time with specific model
  python main.py realtime --model small

  # Batch transcription
  python main.py batch audio.mp3

  # Batch with output file
  python main.py batch audio.mp3 -o transcript.txt

  # Batch with specific model
  python main.py batch audio.mp3 --model medium

  # Verify setup
  python main.py verify
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Real-time mode
    realtime_parser = subparsers.add_parser(
        'realtime',
        help='Real-time transcription from microphone with NeMo'
    )
    realtime_parser.add_argument(
        '-m', '--model',
        help='NeMo model name (overrides config.json)'
    )

    # Batch mode
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch transcription of audio file with NeMo'
    )
    batch_parser.add_argument(
        'audio_file',
        help='Path to audio file (mp3, wav, etc.)'
    )
    batch_parser.add_argument(
        '-o', '--output',
        help='Output transcript file (auto-generated if not specified)'
    )
    batch_parser.add_argument(
        '-m', '--model',
        help='NeMo model name (overrides config.json)'
    )

    # Serve mode (multi-user web UI)
    serve_parser = subparsers.add_parser(
        'serve',
        help='Run the multi-user web server (browser-based realtime transcription)'
    )
    serve_parser.add_argument(
        '--host', default='127.0.0.1',
        help='Bind host (default: 127.0.0.1 — localhost only, mic permission works). '
             'Use 0.0.0.0 for LAN access (mic requires HTTPS in that case).'
    )
    serve_parser.add_argument(
        '--port', type=int, default=8000,
        help='Bind port (default: 8000)'
    )
    serve_parser.add_argument(
        '--reload', action='store_true',
        help='Auto-reload on code changes (development)'
    )

    # Benchmark mode
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run a labeled dataset through an ASR backend; produces WER + gap report'
    )
    benchmark_parser.add_argument(
        '--dataset', default='librispeech', choices=['librispeech'],
        help='Benchmark dataset (default: librispeech)'
    )
    benchmark_parser.add_argument(
        '--split', default='test-clean',
        choices=['test-clean', 'test-other'],
        help='Dataset split (default: test-clean)'
    )
    benchmark_parser.add_argument(
        '--backend', default=None,
        choices=['nemo', 'qwen', 'gemini', 'google-stt', 'whisper'],
        help='Override the ASR backend from config.json'
    )
    benchmark_parser.add_argument(
        '-m', '--model', default=None,
        help='Override the model id (uses backend\'s realtime model if not set)'
    )
    benchmark_parser.add_argument(
        '--limit', type=int, default=None,
        help='Cap the number of samples evaluated (default: all)'
    )
    benchmark_parser.add_argument(
        '-o', '--output', default=None,
        help='Output JSON path (a sibling .md is also written). Auto-named if not given.'
    )
    benchmark_parser.add_argument(
        '--no-cache', action='store_true',
        help='Skip the per-sample transcript cache (force re-transcription)'
    )
    benchmark_parser.add_argument(
        '--allow-download', action='store_true',
        help='Allow downloading the dataset on first run (~350 MB for LibriSpeech)'
    )
    benchmark_parser.add_argument(
        '--confirm-paid', action='store_true',
        help='Required to run cloud backends (gemini, google-stt) over more than 20 samples'
    )

    # Verify mode
    subparsers.add_parser(
        'verify',
        help='Verify system setup and dependencies'
    )

    args = parser.parse_args()

    # Show help if no command
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Route to appropriate mode
    try:
        if args.command == 'realtime':
            realtime_mode(args)
        elif args.command == 'batch':
            batch_mode(args)
        elif args.command == 'serve':
            serve_mode(args)
        elif args.command == 'benchmark':
            benchmark_mode(args)
        elif args.command == 'verify':
            sys.exit(verify_setup())
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
