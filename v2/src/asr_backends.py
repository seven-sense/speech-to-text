"""
Pluggable ASR backends. Supports NVIDIA NeMo, Qwen3-ASR (local), and Gemini
(cloud API). The active backend is selected by `asr_backend` in config.json.
"""

import io
import os
import wave
import warnings
import numpy as np
import torch

from .logger import get_logger

log = get_logger(__name__)

warnings.filterwarnings('ignore')
os.environ.setdefault('NEMO_LOG_LEVEL', 'ERROR')


def pick_device():
    """Return the best available torch device name."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class ASRBackend:
    """Abstract ASR backend. Subclasses implement transcribe()."""

    name = "base"
    device = "cpu"

    def transcribe(self, audio_data: np.ndarray) -> str:
        raise NotImplementedError


class NemoASRBackend(ASRBackend):
    """NVIDIA NeMo ASR backend."""

    name = "nemo"

    def __init__(self, model_name: str, device: str, batch_size: int = 8):
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise ImportError(
                "NeMo is not installed. Install with: pip install nemo_toolkit[asr]"
            ) from e

        import logging
        logging.getLogger('nemo_logger').setLevel(logging.ERROR)

        log.info("Loading NeMo ASR model: %s", model_name)
        print(f"Loading NeMo ASR model: {model_name}...")
        print("(This may take a minute on first run...)")

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        if device == 'cuda':
            self.model = self.model.cuda()
            log.info("NeMo ASR running on CUDA GPU")
            print("[OK] Using CUDA GPU for NeMo ASR")
        elif device == 'mps':
            self.model = self.model.to('mps')
            log.info("NeMo ASR running on MPS")
            print("[OK] Using MPS (Apple Silicon) for NeMo ASR")
        else:
            log.info("NeMo ASR running on CPU (slow)")
            print("[OK] Using CPU for NeMo ASR (slower)")

    def transcribe(self, audio_data: np.ndarray) -> str:
        if len(audio_data) == 0:
            return ""
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        try:
            transcriptions = self.model.transcribe(
                [audio_data],
                batch_size=self.batch_size,
                verbose=False,
            )
            if not transcriptions:
                return ""
            r = transcriptions[0]
            if hasattr(r, 'text'):
                return r.text.strip()
            if isinstance(r, str):
                return r.strip()
            return str(r).strip()
        except Exception as e:
            log.exception("NeMo transcription failed: %s", e)
            return ""


class QwenASRBackend(ASRBackend):
    """Qwen3-ASR backend using the official `qwen-asr` package."""

    name = "qwen"

    def __init__(self, model_name: str, device: str, sample_rate: int = 16000,
                 max_new_tokens: int = 256, language: str | None = None):
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise ImportError(
                "Qwen3-ASR requires the `qwen-asr` package. Install with: "
                "pip install -U qwen-asr"
            ) from e

        # Quiet transformers' chatty per-generation messages
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass

        log.info("Loading Qwen3-ASR model: %s", model_name)
        print(f"Loading Qwen3-ASR model: {model_name}...")
        print("(First download pulls weights from HuggingFace)")

        # Pick dtype — bf16 on CUDA, fp32 elsewhere (MPS bf16 support is patchy).
        if device == 'cuda':
            dtype = torch.bfloat16
            device_map = "cuda:0"
        elif device == 'mps':
            dtype = torch.float32
            device_map = "mps"
        else:
            dtype = torch.float32
            device_map = "cpu"

        self.model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=8,
            max_new_tokens=max_new_tokens,
        )
        self.device = device
        self.sample_rate = sample_rate
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.language = language  # None = auto-detect

        if device == 'cuda':
            log.info("Qwen3-ASR running on CUDA GPU")
            print("[OK] Using CUDA GPU for Qwen3-ASR")
        elif device == 'mps':
            log.info("Qwen3-ASR running on MPS")
            print("[OK] Using MPS (Apple Silicon) for Qwen3-ASR")
        else:
            log.info("Qwen3-ASR running on CPU")
            print("[OK] Using CPU for Qwen3-ASR")

    def transcribe(self, audio_data: np.ndarray) -> str:
        if len(audio_data) == 0:
            return ""
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        audio_data = audio_data.astype(np.float32)

        try:
            results = self.model.transcribe(
                audio=(audio_data, self.sample_rate),
                language=self.language,
            )
            if not results:
                return ""
            text = getattr(results[0], "text", None)
            return (text or "").strip()
        except Exception as e:
            log.exception("Qwen3-ASR transcription failed: %s", e)
            return ""


class WhisperASRBackend(ASRBackend):
    """
    OpenAI Whisper via faster-whisper (CTranslate2 backend).

    Local. ~99 languages. Models range from `tiny` (39 MB) to `large-v3`
    (~1.5 GB). On Apple Silicon we run CPU + int8 (CTranslate2 has no MPS
    backend); on Linux+NVIDIA we use CUDA + float16.

    Models are downloaded from HuggingFace on first use to
    ~/.cache/huggingface (or /root/.cache inside Docker — already cached
    by the model-cache volume).
    """

    name = "whisper"

    def __init__(self, model_name: str, device: str,
                 sample_rate: int = 16000,
                 language: str | None = None,
                 beam_size: int = 5,
                 vad_filter: bool = True):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "Whisper ASR requires `faster-whisper`. Install with: "
                "pip install -U faster-whisper"
            ) from e

        # CTranslate2 device + dtype selection.
        # - CUDA: float16 (fast, accurate)
        # - MPS: not supported by CTranslate2; fall back to CPU int8
        # - CPU: int8 quantized — small model size, fast inference
        if device == "cuda":
            ct2_device = "cuda"
            compute_type = "float16"
        else:
            # MPS or CPU both run on CPU here. int8 quantization keeps
            # large-v3 memory usage around 800 MB instead of ~3 GB.
            ct2_device = "cpu"
            compute_type = "int8"

        log.info("Loading Whisper model: %s (device=%s, compute_type=%s)",
                 model_name, ct2_device, compute_type)
        print(f"Loading Whisper model: {model_name}...")
        print("(First run downloads weights from HuggingFace)")

        self.model = WhisperModel(
            model_name,
            device=ct2_device,
            compute_type=compute_type,
            # download_root=None lets HF caching apply
        )
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.language = language  # None = auto-detect
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        if device == "cuda":
            log.info("Whisper running on CUDA GPU")
            print("[OK] Using CUDA GPU for Whisper")
        elif device == "mps":
            # We told CT2 cpu — log honestly so the user knows.
            log.info("Whisper running on CPU (CTranslate2 has no MPS support)")
            print("[OK] Using CPU for Whisper "
                  "(CTranslate2 has no Apple-Silicon MPS backend)")
        else:
            log.info("Whisper running on CPU")
            print("[OK] Using CPU for Whisper")

    def transcribe(self, audio_data: np.ndarray) -> str:
        if len(audio_data) == 0:
            return ""
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        audio_data = audio_data.astype(np.float32)

        try:
            segments, _info = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                # condition_on_previous_text=False reduces hallucination on
                # chunked input — important for our streaming pipeline where
                # each call has only ~2-4s of context.
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text
        except Exception as e:
            log.exception("Whisper transcription failed: %s", e)
            return ""


class GeminiASRBackend(ASRBackend):
    """
    Gemini ASR backend (cloud). Uses google-genai to call Gemini multimodal
    models with the audio chunk + a transcription prompt.

    Audio is sent as 16-bit PCM wrapped in a WAV header. No model weights are
    downloaded — every call is a network round-trip (~1–3s per chunk).
    """

    name = "gemini"

    def __init__(self, model_name: str, api_key: str | None,
                 sample_rate: int = 16000,
                 prompt: str | None = None,
                 language: str | None = None):
        try:
            from google import genai  # noqa: F401  (validates install)
        except ImportError as e:
            raise ImportError(
                "Gemini ASR requires the `google-genai` package. Install with: "
                "pip install -U google-genai"
            ) from e

        if not api_key:
            raise RuntimeError(
                "Gemini ASR requires an API key. Set GEMINI_API_KEY (or "
                "GOOGLE_API_KEY) in your .env, or pass it via the config."
            )

        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = "cloud"  # not a torch device; here for ASRBackend interface
        self.language = language

        if prompt is None:
            base = "Transcribe the spoken audio verbatim. Output only the transcript text — no commentary, no timestamps, no speaker labels."
            if language:
                base += f" The audio is in {language}."
            prompt = base
        self.prompt = prompt

        log.info("Gemini ASR ready (model=%s, lang=%s)",
                 model_name, language or "auto")
        print(f"[OK] Using Gemini API for ASR (model: {model_name})")

    @staticmethod
    def _pcm_float32_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
        """Convert mono float32 PCM [-1, 1] to 16-bit WAV bytes (in memory)."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
            wf.writeframes(int16.tobytes())
        return buf.getvalue()

    def transcribe(self, audio_data: np.ndarray) -> str:
        if len(audio_data) == 0:
            return ""
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        audio_data = audio_data.astype(np.float32)

        try:
            from google.genai import types
            wav_bytes = self._pcm_float32_to_wav_bytes(
                audio_data, self.sample_rate
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    self.prompt,
                ],
            )
            text = (response.text or "").strip()
            # Gemini sometimes adds quoting or "Transcript:" prefixes — strip them.
            for prefix in ("Transcript:", "Transcription:"):
                if text.lower().startswith(prefix.lower()):
                    text = text[len(prefix):].strip()
            text = text.strip('"').strip("'").strip()
            return text
        except Exception as e:
            log.exception("Gemini transcription failed: %s", e)
            return ""


class GoogleSTTBackend(ASRBackend):
    """
    Google Cloud Speech-to-Text v2 backend (Chirp / latest_long / etc).

    Auth: relies on Application Default Credentials. Set
    GOOGLE_APPLICATION_CREDENTIALS to point at a service-account JSON file.
    Set GOOGLE_CLOUD_PROJECT (or pass via config) to specify the GCP project.

    No model weights are downloaded; every call is a network round-trip.
    Free tier: ~60 min/month. Pricing scales by the audio model.
    """

    name = "google-stt"

    # Models in the Cloud STT v2 catalog and where they're available.
    # chirp_3 is only deployed at "global"; the others are only deployed at
    # regional locations. There's no single location that supports all of
    # them, so we auto-pick based on the model name.
    _GLOBAL_ONLY_MODELS = {"chirp_3"}
    _REGIONAL_ONLY_MODELS = {"chirp_2", "chirp", "latest_long", "latest_short"}

    def __init__(self, model_name: str, project_id: str | None,
                 location: str = "us-central1",
                 sample_rate: int = 16000,
                 language_codes: list[str] | None = None):

        # Auto-correct the location based on the chosen model. Cloud STT v2
        # deploys different models in different locations; chirp_3 is global-
        # only, the rest are regional-only. We honor the configured location
        # but flip it when the model demands otherwise.
        original_location = location
        loc_lower = (location or "").lower()
        if model_name in self._GLOBAL_ONLY_MODELS and loc_lower != "global":
            location = "global"
        elif model_name in self._REGIONAL_ONLY_MODELS and loc_lower == "global":
            location = "us-central1"
        if location != original_location:
            log.info(
                "google-stt: model %s requires location %s — overriding "
                "configured %s",
                model_name, location, original_location,
            )
        try:
            from google.cloud import speech_v2  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Google Cloud Speech-to-Text requires `google-cloud-speech`. "
                "Install with: pip install -U google-cloud-speech"
            ) from e

        if not project_id:
            raise RuntimeError(
                "Google Cloud STT requires a project id. Set "
                "GOOGLE_CLOUD_PROJECT in your .env or models.google-stt.project "
                "in config.json."
            )
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            log.warning(
                "GOOGLE_APPLICATION_CREDENTIALS is not set. Google Cloud STT "
                "will fail unless ADC are otherwise available."
            )

        from google.cloud import speech_v2
        from google.api_core.client_options import ClientOptions

        # Speech-to-Text v2 uses regional endpoints. The client must connect
        # to <region>-speech.googleapis.com when the recognizer's location is
        # regional. Only "global" uses the default endpoint.
        client_opts = None
        if location and location.lower() != "global":
            client_opts = ClientOptions(
                api_endpoint=f"{location}-speech.googleapis.com"
            )

        self._speech_v2 = speech_v2
        self.client = speech_v2.SpeechClient(client_options=client_opts)
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.sample_rate = sample_rate
        self.language_codes = language_codes or ["en-US"]
        self.device = "cloud"
        self.recognizer = (
            f"projects/{project_id}/locations/{location}/recognizers/_"
        )
        log.info(
            "Google Cloud STT ready (model=%s, location=%s, project=%s, langs=%s)",
            model_name, location, project_id, ",".join(self.language_codes),
        )
        print(f"[OK] Using Google Cloud STT (model: {model_name}, location: {location})")

    @staticmethod
    def _pcm_float32_to_linear16(pcm: np.ndarray) -> bytes:
        """Convert mono float32 PCM [-1, 1] to LINEAR16 (16-bit PCM) bytes."""
        int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
        return int16.tobytes()

    def transcribe(self, audio_data: np.ndarray) -> str:
        if len(audio_data) == 0:
            return ""
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        audio_data = audio_data.astype(np.float32)

        try:
            from google.cloud.speech_v2.types import cloud_speech

            cfg = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=1,
                ),
                language_codes=self.language_codes,
                model=self.model_name,
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                ),
            )
            request = cloud_speech.RecognizeRequest(
                recognizer=self.recognizer,
                config=cfg,
                content=self._pcm_float32_to_linear16(audio_data),
            )
            response = self.client.recognize(request=request)
            parts = []
            for result in response.results:
                if result.alternatives:
                    parts.append(result.alternatives[0].transcript)
            return " ".join(parts).strip()
        except Exception as e:
            log.exception("Google Cloud STT transcription failed: %s", e)
            return ""


def build_asr_backend(config, use_realtime_model: bool = False,
                      model_override: str | None = None) -> ASRBackend:
    """Construct the ASR backend named by config.asr_backend."""
    backend = (config.asr_backend or "nemo").lower()

    if backend == "nemo":
        model_name = model_override or (
            config.nemo_realtime_model if use_realtime_model
            else config.nemo_batch_model
        )
    elif backend == "qwen":
        model_name = model_override or (
            config.qwen_realtime_model if use_realtime_model
            else config.qwen_batch_model
        )
    elif backend == "gemini":
        model_name = model_override or (
            config.gemini_realtime_model if use_realtime_model
            else config.gemini_batch_model
        )
    elif backend == "google-stt":
        model_name = model_override or (
            config.google_stt_realtime_model if use_realtime_model
            else config.google_stt_batch_model
        )
    elif backend == "whisper":
        model_name = model_override or (
            config.whisper_realtime_model if use_realtime_model
            else config.whisper_batch_model
        )
    else:
        raise ValueError(
            f"Unknown asr_backend: {backend!r}. "
            "Valid options: 'nemo', 'qwen', 'gemini', 'google-stt', 'whisper'."
        )

    return make_asr_backend(backend, model_name, config)


def make_asr_backend(backend: str, model_name: str, config) -> ASRBackend:
    """
    Build a specific ASR backend by name + model id, ignoring
    config.asr_backend. Used by the web server's per-session model pool.
    """
    backend = (backend or "").lower()
    device = pick_device()

    if backend == "nemo":
        return NemoASRBackend(
            model_name, device,
            batch_size=config.nemo_params["batch_size"],
        )
    if backend == "qwen":
        return QwenASRBackend(
            model_name, device,
            sample_rate=config.sample_rate,
            language=config.qwen_language,
        )
    if backend == "gemini":
        return GeminiASRBackend(
            model_name,
            api_key=config.gemini_api_key,
            sample_rate=config.sample_rate,
            language=config.gemini_language,
        )
    if backend == "google-stt":
        return GoogleSTTBackend(
            model_name,
            project_id=config.google_stt_project,
            location=config.google_stt_location,
            sample_rate=config.sample_rate,
            language_codes=config.google_stt_languages,
        )
    if backend == "whisper":
        return WhisperASRBackend(
            model_name, device,
            sample_rate=config.sample_rate,
            language=config.whisper_language,
            beam_size=config.whisper_beam_size,
            vad_filter=config.whisper_vad_filter,
        )
    raise ValueError(
        f"Unknown ASR backend: {backend!r}. "
        "Valid options: 'nemo', 'qwen', 'gemini', 'google-stt', 'whisper'."
    )
