# Speech Test

This project consists of two Python scripts that facilitate voice-based interaction using WebSockets, Speech-to-Text (STT), and Text-to-Speech (TTS) functionalities.

## Components

### 1. `test.py` (WebSocket Server)
- Initializes a WebSocket server to handle real-time voice interactions.
- Implements STT using Faster Whisper for transcription.
- Generates AI-based responses using the Groq API.
- Converts text responses to speech using the ElevenLabs API.
- Supports VAD (Voice Activity Detection) for noise reduction.
- Manages client connections and processes audio messages.

### 2. `client.py` (WebSocket Client)
- Connects to the WebSocket server.
- Records and sends audio input to the server.
- Receives and plays back the AI-generated response.
- Uses sounddevice for audio recording and ElevenLabs for playback.

## Features
- **Real-time speech interaction** through WebSockets.
- **AI-powered responses** using Groq API.
- **Enhanced speech processing** with Faster Whisper STT and ElevenLabs TTS.
- **Efficient client-server communication** using base64-encoded audio transmission.

## Additional Notes
- AI tools were used for optimizing WebSocket connections and troubleshooting the code.

## Installation & Usage
1. Install dependencies using `pip install -r requirements.txt`.
2. Set up environment variables for `GROQ_API_KEY` and `TTS_API_KEY`.
3. Run the server: `python3 test.py`.
4. Start the client: `python3 client.py`.
5. Follow on-screen instructions to interact using voice.

## Future Enhancements
- Support for additional languages.
- Improved VAD filtering.
- UI integration for a seamless experience.

---
This project enables real-time, AI-driven voice interaction for various applications like virtual assistants and accessibility tools.

### npy_assnment.ipynb 
Solves the shape disfiguration problem using PCA
