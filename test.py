#!/usr/bin/env python3
import asyncio
import websockets
import json
import logging
import os
import base64
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VoiceChatServer")

# Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")

if not GROQ_API_KEY or not TTS_API_KEY:
    logger.error("Missing API keys in environment variables.")
    exit(1)

TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

# Speech-to-Text using Faster Whisper
class SpeechToText:
    def __init__(self, model_size="tiny"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    async def transcribe(self, audio_path: Path) -> str:
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            segments, _ = await asyncio.to_thread(self.model.transcribe, str(audio_path), beam_size=1, vad_filter=True)
            transcription = " ".join(segment.text for segment in segments)
            logger.info(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

# AI Response Generation via Groq API
class AIModel:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "mixtral-8x7b-32768"

    async def generate_response(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": text}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.base_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Groq API error {response.status}: {await response.text()}")
                        return "Sorry, I couldn't process your request."
                    response_data = await response.json()
                    response_text = response_data["choices"][0]["message"]["content"]
                    logger.info(f"AI Response: {response_text}")
                    return response_text
            except Exception as e:
                logger.error(f"Groq API network error: {e}")
                return "Network error occurred."

# Text-to-Speech using ElevenLabs API
class TextToSpeech:
    def __init__(self, api_key: str):
        self.client = ElevenLabs(api_key=api_key)

    async def synthesize(self, text: str, output_path: Path) -> bool:
        try:
            logger.info(f"Synthesizing TTS for text: {text}")
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )
            with open(output_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)
            logger.info(f"TTS audio saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return False

# WebSocket Server
class VoiceChatServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.stt = SpeechToText()
        self.ai_model = AIModel(GROQ_API_KEY)
        self.tts = TextToSpeech(TTS_API_KEY)

    async def handle_client(self, websocket):
        client_id = id(websocket)
        path = getattr(websocket, "path", "unknown")
        logger.info(f"Client connected: {client_id}, path: {path}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "audio" and data.get("audio_data"):
                        audio_file = TEMP_AUDIO_DIR / f"client_{client_id}.wav"
                        with open(audio_file, "wb") as f:
                            f.write(base64.b64decode(data["audio_data"]))
                        logger.info(f"Received audio file: {audio_file}")

                        transcription = await self.stt.transcribe(audio_file)
                        await websocket.send(json.dumps({"type": "transcription", "text": transcription}))

                        response_text = await self.ai_model.generate_response(transcription)
                        await websocket.send(json.dumps({"type": "response", "text": response_text}))

                        tts_output = TEMP_AUDIO_DIR / f"client_{client_id}.mp3"
                        if await self.tts.synthesize(response_text, tts_output):
                            with open(tts_output, "rb") as f:
                                audio_b64 = base64.b64encode(f.read()).decode()
                            await websocket.send(json.dumps({"type": "audio", "audio_data": audio_b64}))
                        else:
                            await websocket.send(json.dumps({"type": "error", "message": "TTS synthesis failed."}))

                        # Clean up temporary files
                        audio_file.unlink(missing_ok=True)
                        tts_output.unlink(missing_ok=True)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON message received.")
                except KeyError as e:
                    logger.error(f"Missing key in message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")

    async def start(self):
        logger.info(f"Server running on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  

if __name__ == "__main__":
    server = VoiceChatServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")