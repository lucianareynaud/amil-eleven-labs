# Voice-to-URA Pipeline

This directory contains the backend implementation of the Voice-to-URA pipeline, which transforms audio recordings into URA-ready voice prompts using a combination of:

1. **Whisper** for speech-to-text transcription
2. **Local LLM** (Mistral via LLaMA.cpp) for URA-style text rewriting
3. **ElevenLabs API** for high-quality voice synthesis

## Architecture

The pipeline follows these steps:

1. **Audio Ingestion**: Upload audio file via FastAPI endpoint
2. **Transcription**: Whisper model transcribes speech to text
3. **URA Rewriting**: Mistral LLM reformats text to URA standards with SSML markup
4. **Text Processing**: Validates, cleans, and annotates text with SSML
5. **Voice Synthesis**: ElevenLabs API generates professional audio
6. **Result Delivery**: Returns transcription, URA text, and audio file

## Setup

1. Ensure you have all dependencies installed:
   ```
   pip install -r ../requirements.txt
   ```

2. Create a `.env` file with your configuration (see `env.example` for reference)
   ```
   ELEVENLABS_API_KEY=your_key_here
   WHISPER_MODEL=tiny
   MISTRAL_MODEL_PATH=/app/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
   MISTRAL_N_CTX=512
   ```

3. Make sure you have downloaded the Mistral model:
   ```
   ./download-mistral.sh
   ```

## Usage

### Running the Service

Start the service with:
```
python run_ura_service.py
```

This will start the FastAPI server on port 8000 by default.

### Testing ElevenLabs Integration

Test that your ElevenLabs API key works:
```
python test_elevenlabs.py
```

### API Endpoints

- **GET /health**: Check service health
- **POST /process-audio**: Process audio file through pipeline
- **GET /download/{request_id}**: Download generated audio file

## Example Usage

```python
import requests

# Process audio file
with open("test_audio.webm", "rb") as f:
    files = {"file": ("recording.webm", f, "audio/webm")}
    response = requests.post("http://localhost:8000/process-audio", files=files)
    
result = response.json()
print(f"Transcript: {result['transcript']}")
print(f"URA Text: {result['ura_text']}")
print(f"Output file: {result['output_file']}")
``` 