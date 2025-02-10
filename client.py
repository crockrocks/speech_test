#!/usr/bin/env python3
import asyncio
import websockets
import json
import logging
import base64
import sounddevice as sd
import numpy as np
import wave
from pathlib import Path
from io import BytesIO
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VoiceChatClient")

SERVER_URI = "ws://localhost:8765"
AUDIO_SAMPLE_RATE = 16000
CHANNELS = 1
TEMP_AUDIO_FILE = Path("client_recording.wav")

async def record_audio(duration=5, filename=TEMP_AUDIO_FILE):
    logger.info("Recording audio...")
    loop = asyncio.get_event_loop()
    
    # Start recording in a non-blocking way
    recording = np.zeros((int(duration * AUDIO_SAMPLE_RATE), CHANNELS), dtype=np.int16)
    stream = sd.InputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=int(AUDIO_SAMPLE_RATE * 0.5),  # 500ms blocks
    )
    
    with stream:
        await loop.run_in_executor(None, stream.start)
        frames = []
        for _ in range(int(duration * 2)):  # 0.5s per block
            data, _ = await loop.run_in_executor(None, stream.read, int(AUDIO_SAMPLE_RATE * 0.5))
            frames.append(data)
        recording = np.concatenate(frames, axis=0)
    
    # Save recording asynchronously
    def save_wav():
        with wave.open(str(filename), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(recording.tobytes())
    
    await loop.run_in_executor(None, save_wav)
    return filename

async def play_audio_from_base64(audio_b64):
    try:
        audio_bytes = base64.b64decode(audio_b64)
        audio = AudioSegment.from_mp3(BytesIO(audio_bytes))
        samples = np.array(audio.get_array_of_samples())
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
        else:
            samples = samples.reshape((-1, 1))
        
        def play_sync():
            sd.play(samples, samplerate=audio.frame_rate)
            sd.wait()
        
        await asyncio.to_thread(play_sync)
    except Exception as e:
        logger.error(f"Audio playback error: {e}")

class VoiceChatClient:
    def __init__(self, uri=SERVER_URI):
        self.uri = uri
        self.websocket = None

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info("Connected to WebSocket server.")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    async def send_audio(self, filename):
        try:
            with open(filename, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            await self.websocket.send(json.dumps({"type": "audio", "audio_data": audio_b64}))
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    async def receive_messages(self):
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get("type") == "audio":
                    logger.info("Playing received TTS audio.")
                    await play_audio_from_base64(data["audio_data"])
                elif data.get("type") == "transcription":
                    logger.info(f"Transcription: {data['text']}")
                elif data.get("type") == "response":
                    logger.info(f"AI Response: {data['text']}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server.")

    async def run(self):
        await self.connect()
        receive_task = asyncio.create_task(self.receive_messages())
        
        try:
            while True:
                # Async user input
                user_input = await asyncio.to_thread(
                    input, "Press Enter to record (or type 'quit' to exit): "
                )
                if user_input.strip().lower() == 'quit':
                    break
                
                recorded_file = await record_audio()
                await self.send_audio(recorded_file)
                recorded_file.unlink(missing_ok=True)
        finally:
            receive_task.cancel()
            await self.websocket.close()

async def main():
    client = VoiceChatClient()
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client shutting down.")