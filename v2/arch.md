# Speech-to-Text v2 Architecture Diagrams

This document separates the project's codebase architecture into dedicated, easy-to-read diagrams for each execution pipeline.

---

## 1. Web Real-Time Server Flow (`python main.py serve`)
Handles live microphone streaming from multiple concurrent web users.

```mermaid
flowchart LR
    %% Inputs
    browser[Browser Mic: Web Audio API]
    
    %% Server Layer
    web_server[server.py FastAPI]
    session[Session Manager]
    
    %% Audio Utilities
    vad[VADEndpointer RMS]
    rolling_writer[RollingMP3Writer]
    
    %% Adapters
    subgraph adapters [ASR Stream Adapters]
        nemo_str[NemoStreaming]
        google_grpc[GoogleSTTStream]
        gemini_live[GeminiLive]
        chunk_adap[ChunkedASR]
    end
    
    %% ASR Execution
    model_lock[model_lock lock]
    asr_engines[ASR Engines: NeMo/Qwen/Whisper/Gemini/Google STT]
    
    %% Outputs
    transcripts[transcripts/web_*.txt]
    saved_audio[audio/web_*.mp3]

    %% Connections
    browser -->|WebSockets: float32 PCM| web_server
    web_server --> session
    session --> rolling_writer
    session --> vad
    session --> adapters
    
    adapters --> model_lock
    model_lock --> asr_engines
    
    session --> transcripts
    rolling_writer --> saved_audio
```

---

## 2. Local CLI Real-Time Flow (`python main.py realtime`)
Transcribes raw voice from the local computer mic directly to the command line.

```mermaid
flowchart LR
    %% Inputs
    user_mic[Mic Input]
    
    %% Controller
    rt_trans[RealtimeTrans]
    
    %% Storage FIFO
    audio_buf[AudioBuffer FIFO]
    
    %% ASR Execution
    model_lock[model_lock lock]
    asr_engines[ASR Engines: NeMo/Qwen/Whisper/Gemini/Google STT]
    
    %% Outputs
    transcripts[transcripts/realtime_*.txt]
    console[Console Output]

    %% Connections
    user_mic -->|sounddevice stream| rt_trans
    rt_trans -->|Feed audio| audio_buf
    audio_buf -->|Consume & VAD/Overlap slice| rt_trans
    
    rt_trans --> model_lock
    model_lock --> asr_engines
    
    rt_trans --> transcripts
    rt_trans --> console
```

---

## 3. Batch File Processing Flow (`python main.py batch <file>`)
Transcribes pre-recorded files offline, using chunking to maintain a low RAM footprint.

```mermaid
flowchart LR
    %% Inputs
    audio_files[Audio Files]
    
    %% Controller & Slicer
    batch_trans[BatchTrans]
    file_chunk[chunk_audio_file]
    pydub_load[audio_utils: load_audio / float32]
    
    %% ASR Execution
    model_lock[model_lock lock]
    asr_engines[ASR Engines: NeMo/Qwen/Whisper/Gemini/Google STT]
    
    %% Outputs
    transcripts[transcripts/<file_name>_*.txt]

    %% Connections
    audio_files --> batch_trans
    batch_trans --> file_chunk
    file_chunk --> pydub_load
    
    pydub_load -->|Float32 array| batch_trans
    batch_trans --> model_lock
    model_lock --> asr_engines
    
    batch_trans --> transcripts
```

---

## 4. Evaluator / Benchmark Flow (`python main.py benchmark`)
Downloads datasets and evaluates model transcription accuracy (WER/CER).

```mermaid
flowchart LR
    %% Inputs
    hf_dataset[HuggingFace Hub]
    
    %% Controllers
    datasets_loader[datasets.py loader]
    evaluator[Evaluator]
    pydub_load[audio_utils: load_audio / float32]
    
    %% Cache
    cache[benchmark_cache/]
    
    %% ASR Execution
    model_lock[model_lock lock]
    asr_engines[ASR Engines: NeMo/Qwen/Whisper/Gemini/Google STT]
    
    %% Scorer
    jiwer[jiwer metrics]
    report_writer[report.py]
    
    %% Outputs
    reports[reports/*.json & reports/*.md]

    %% Connections
    hf_dataset --> datasets_loader
    datasets_loader -->|Iterate Samples| evaluator
    
    evaluator -->|Check cache JSON| cache
    cache -.->|Cache Hit: read text| evaluator
    
    evaluator -->|Cache Miss: load audio| pydub_load
    pydub_load --> evaluator
    evaluator --> model_lock
    model_lock --> asr_engines
    
    evaluator -->|Reference vs Hypothesis| jiwer
    jiwer --> report_writer
    report_writer --> reports
```