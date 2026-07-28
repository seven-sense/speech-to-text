# Real-Time Multi-Model Speech-to-Text System

## Overview

A real-time Speech-to-Text (STT) application that supports multiple Automatic Speech Recognition (ASR) backends, including Qwen3-ASR, Whisper, and NVIDIA NeMo. The system captures audio from a microphone, processes speech in real time, and generates text transcriptions through a web-based interface and backend API.

This project was explored and enhanced as part of an internship at Granite River Labs (GRL), focusing on understanding real-world AI system architecture, model integration, and debugging complex codebases.

---

## Features

* Real-time speech transcription from microphone input
* Multiple ASR backend support:

  * Qwen3-ASR
  * OpenAI Whisper
  * NVIDIA NeMo
* Web-based user interface
* FastAPI backend
* Configurable model selection
* Logging and monitoring support
* Modular architecture for adding new ASR models

---

## Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### AI Models

* Qwen3-ASR
* OpenAI Whisper
* NVIDIA NeMo

### Audio Processing

* SoundDevice
* NumPy

### Deployment & Environment

* Docker
* Docker Compose

---

## System Architecture

```text
Microphone Input
        ↓
Audio Processing Layer
        ↓
Selected ASR Backend
(Whisper / NeMo / Qwen3-ASR)
        ↓
Transcription Engine
        ↓
FastAPI Backend
        ↓
Web Interface
```

---

## Project Structure

```text
stt-project/
│
├── backend/
├── frontend/
├── models/
├── config/
├── logs/
├── docker/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd stt-project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running with Docker

```bash
docker-compose up --build
```

---

## Running Locally

```bash
uvicorn server:app --reload
```

---

## Configuration

The application allows switching between multiple ASR backends through configuration files.

Example:

```json
{
  "asr_backend": "qwen"
}
```

Supported values:

* qwen
* whisper
* nemo

---

## Supported Models

| Model       | Description                                           |
| ----------- | ----------------------------------------------------- |
| Qwen3-ASR   | Context-aware multilingual ASR model                  |
| Whisper     | General-purpose multilingual speech recognition model |
| NVIDIA NeMo | Conversational AI and ASR toolkit                     |

---

## Learning Outcomes

* Understanding real-time ASR pipelines
* Debugging and configuring complex AI applications
* Working with multiple speech recognition backends
* Understanding system architecture and data flow
* Performance monitoring and resource analysis
* Integrating AI models into production-style applications

---

## Future Improvements

* Speaker diarization
* Streaming transcription using WebSockets
* GPU performance optimization
* Automatic language detection
* Transcription export functionality
* Meeting summarization integration

---

## License

This project is intended for educational and research purposes.
