from __future__ import annotations
import io
import base64
import asyncio
import threading
from typing import Callable, Awaitable
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Constants for audio processing
CHUNK_LENGTH_S = 0.05  # 50ms
SAMPLE_RATE = 24000
CHANNELS = 1

def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
    # Load audio using pydub and resample to 24kHz, mono, PCM16 format
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    pcm_audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2).raw_data
    return pcm_audio

class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(
            callback=self.callback,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16,
            blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
        )
        self.playing = False
        self._frame_count = 0

    def callback(self, outdata, frames, time, status):  # noqa
        with self.lock:
            data = np.empty(0, dtype=np.int16)
            # Fill the output buffer using queued PCM data
            while len(data) < frames and self.queue:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])
            self._frame_count += len(data)
            # In case not enough data, pad with zeros
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=np.int16)))
        outdata[:] = data.reshape(-1, 1)

    def add_data(self, data: bytes):
        with self.lock:
            np_data = np.frombuffer(data, dtype=np.int16)
            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        self.playing = True
        self.stream.start()

    def stop(self):
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue.clear()

    def terminate(self):
        self.stream.close()

async def send_audio_worker_sounddevice(
    connection,
    should_send: Callable[[], bool] | None = None,
    start_send: Callable[[], Awaitable[None]] | None = None,
):
    sent_audio = False
    read_size = int(SAMPLE_RATE * 0.02)  # 20ms chunks
    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16",
    )
    stream.start()
    try:
        while True:
            if stream.read_available < read_size:
                await asyncio.sleep(0.01)
                continue
            data, _ = stream.read(read_size)
            if should_send() if should_send else True:
                if not sent_audio and start_send:
                    await start_send()
                await connection.send({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode("utf-8")
                })
                sent_audio = True
            elif sent_audio:
                print("Audio done, triggering inference")
                await connection.send({"type": "input_audio_buffer.commit"})
                await connection.send({"type": "response.create", "response": {}})
                sent_audio = False
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

async def receive_audio_events(connection, player: AudioPlayerAsync):
    async for event in connection:
        if event.type == "response.audio_transcript.delta":
            print(f"Transcript: {event.delta}", flush=True)
        elif event.type == "response.audio.delta":
            # Decode incoming base64 audio delta and play it
            buffer = base64.b64decode(event.delta)
            print(f"Received {len(buffer)} bytes of audio data.")
            player.add_data(buffer)
        elif event.type == "response.done":
            break

async def main() -> None:
    url_cognitiveservices = os.getenv("URL_COGNITIVESERVICES")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    # Initialize Azure credentials and the AsyncAzureOpenAI client
    credential = DefaultAzureCredential()
    client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=get_bearer_token_provider(
            credential, f"{url_cognitiveservices}"
        ),
        api_version="2024-10-01-preview",
    )
    player = AudioPlayerAsync()
    try:
        while True:
            async with client.beta.realtime.connect(
                model="gpt-4o-realtime-preview",
            ) as connection:
                # Use both audio and text modalities
                await connection.session.update(session={"modalities": ["audio", "text"]})
                # Start the worker sending audio from sounddevice
                send_task = asyncio.create_task(send_audio_worker_sounddevice(connection))
                # Listen for incoming audio events and transcripts
                await receive_audio_events(connection, player)
                send_task.cancel()
                print("Reload chat of speak...")
    except KeyboardInterrupt:
        print("Ending chat of speak.")
    finally:
        player.terminate()
        await credential.close()

if __name__ == "__main__":
    asyncio.run(main())