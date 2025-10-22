import asyncio
import base64
import json
import os
import signal
import sys
from contextlib import asynccontextmanager

import pyaudio
import websockets
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.stderr.write("Missing OPENAI_API_KEY environment variable.\n")
    sys.exit(1)


MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-mini-transcribe")
INTENT = os.getenv("OPENAI_REALTIME_INTENT", "transcription")
REALTIME_URL = f"wss://api.openai.com/v1/realtime?intent={INTENT}"
SESSION_URL = "https://api.openai.com/v1/realtime/transcription_sessions"

# The streaming ASR API expects 24 kHz, 16-bit PCM mono audio when using pcm16.
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
FRAMES_PER_BUFFER = 2_400  # 0.1s of audio per packet at 24 kHz


class RealtimeTranscriber:
    def __init__(self) -> None:
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )
        self._should_stop = asyncio.Event()
        self._text_buffers: dict[str, list[str]] = {}

    def stop(self) -> None:
        if not self._should_stop.is_set():
            self._should_stop.set()

    async def run(self) -> None:
        try:
            session_token = await self._create_session_token()
            async with self._connect(session_token) as websocket:
                print("Connected to realtime endpoint.", flush=True)
                await self._initialize_session(websocket)
                sender = asyncio.create_task(self._send_audio(websocket))
                receiver = asyncio.create_task(self._consume_events(websocket))
                await self._should_stop.wait()
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
        finally:
            self._stream.stop_stream()
            self._stream.close()
            self._audio.terminate()

    async def _create_session_token(self) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._request_session_token)

    def _request_session_token(self) -> str:
        try:
            response = requests.post(
                SESSION_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "assistants=v2",
                },
                json={},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                detail = exc.response.text
            raise RuntimeError(f"Failed to create transcription session. {detail}") from exc

        token = response.json().get("client_secret", {}).get("value")
        if not token:
            raise RuntimeError("No transcription session token returned.")
        return token

    @asynccontextmanager
    async def _connect(self, session_token: str):
        headers = {
            "Authorization": f"Bearer {session_token}",
            "OpenAI-Beta": "realtime=v1",
            "OpenAI-Intent": INTENT,
        }
        async with websockets.connect(
            REALTIME_URL,
            additional_headers=headers,
            ping_interval=10,
            ping_timeout=20,
            max_size=10 * 1024 * 1024,
        ) as websocket:
            yield websocket

    async def _initialize_session(self, websocket):
        session_update = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": MODEL},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
        }
        await websocket.send(json.dumps(session_update))

    async def _send_audio(self, websocket):
        loop = asyncio.get_running_loop()
        try:
            while not self._should_stop.is_set():
                chunk = await loop.run_in_executor(
                    None,
                    lambda: self._stream.read(
                        FRAMES_PER_BUFFER, exception_on_overflow=False
                    ),
                )
                if not chunk:
                    continue
                if len(chunk) < FRAMES_PER_BUFFER * SAMPLE_WIDTH_BYTES:
                    continue
                encoded = base64.b64encode(chunk).decode("utf-8")
                await websocket.send(
                    json.dumps(
                        {"type": "input_audio_buffer.append", "audio": encoded}
                    )
                )
        except asyncio.CancelledError:
            pass

    async def _consume_events(self, websocket):
        try:
            async for message in websocket:
                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "conversation.item.input_audio_transcription.delta":
                    item_id = event.get("item_id")
                    text = event.get("delta")
                    if item_id and text:
                        self._text_buffers.setdefault(item_id, []).append(text)
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    item_id = event.get("item_id")
                    transcript = event.get("transcript")
                    if not transcript and item_id:
                        transcript = "".join(self._text_buffers.pop(item_id, [])).strip()
                    if item_id in self._text_buffers:
                        self._text_buffers.pop(item_id, None)
                    if transcript:
                        print(transcript.strip(), flush=True)
                elif event_type == "input_audio_buffer.committed":
                    pass
                elif event_type == "input_audio_buffer.cleared":
                    pass
                elif event_type == "session.error":
                    message = event.get("error", {}).get("message")
                    sys.stderr.write(f"[Session Error] {message}\n")
                elif event_type == "end_of_stream":
                    self._text_buffers.clear()
                    self.stop()
                elif event_type == "error":
                    message = event.get("error", {}).get("message", "Unknown error.")
                    sys.stderr.write(f"[Realtime Error] {message}\n")
                else:
                    continue
        except asyncio.CancelledError:
            pass


def main() -> None:
    transcriber = RealtimeTranscriber()

    def _handle_signal(*_):
        transcriber.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except ValueError:
            pass

    try:
        asyncio.run(transcriber.run())
    except KeyboardInterrupt:
        transcriber.stop()


if __name__ == "__main__":
    main()
