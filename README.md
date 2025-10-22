# openai-realtime-transcribe

A command-line client that streams microphone audio to the OpenAI Realtime API using `pyaudio`, printing partial and final transcripts in real time.

The core logic lives in `realtime_transcribe.py`:
- `RealtimeTranscriber.__init__`: sets up the `PyAudio` input stream and control events.
- `_create_session_token()` / `_request_session_token()`: call the REST endpoint to create a transcription session and obtain a short-lived token.
- `_connect()`: establish the Realtime WebSocket connection with that token.
- `_initialize_session()`: send the session configuration (audio format, transcription model, server-side VAD settings).
- `_send_audio()`: asynchronous task that continuously reads PCM audio chunks, base64-encodes them, and streams them to the server.
- `_consume_events()`: asynchronous task that consumes server events, stitches together `delta` chunks, and prints completed transcripts.
- `main()`: wires up signal handlers and runs the transcription loop.

## Requirements
- Python 3.10+
- `portaudio` (macOS/Linux); on Windows install `pip install pipwin && pipwin install pyaudio` to get a prebuilt wheel.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration
Copy `.env.example` to `.env` and fill in your credentials:
```env
OPENAI_API_KEY=sk-xxxx
# OPENAI_REALTIME_MODEL=gpt-4o-mini-transcribe
# OPENAI_REALTIME_INTENT=transcription
```

The default model is `gpt-4o-mini-transcribe`. Switch `OPENAI_REALTIME_MODEL` to `gpt-4o-transcribe` if you prefer higher accuracy at the cost of more latency and price.

## Run
```bash
python realtime_transcribe.py
```

After startup the script immediately begins reading from the microphone, streaming audio to the Realtime endpoint, and printing transcripts as they arrive. Press `Ctrl+C` to stop.
