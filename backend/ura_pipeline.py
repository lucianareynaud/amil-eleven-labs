import os
import uuid
import tempfile
import subprocess
import logging
import time
import asyncio
import requests
import re
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from llama_cpp import Llama
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
MISTRAL_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
MISTRAL_N_CTX = int(os.getenv("MISTRAL_N_CTX", "512"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "0.5"))

# Try to load the prompt template
MISTRAL_PROMPT_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mistral-prompt-template.txt")
MISTRAL_SYSTEM_PROMPT = ""
if os.path.exists(MISTRAL_PROMPT_TEMPLATE_PATH):
    with open(MISTRAL_PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        MISTRAL_SYSTEM_PROMPT = f.read().strip()
    logger.info(f"Loaded system prompt from {MISTRAL_PROMPT_TEMPLATE_PATH}")
else:
    logger.warning(f"Prompt template file not found at {MISTRAL_PROMPT_TEMPLATE_PATH}, using default system prompt")

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "ErXwobaYiN019PkySvjV")  # Antoni (male) for pt-BR
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v1")
ELEVENLABS_STABILITY = float(os.getenv("ELEVENLABS_STABILITY", "0.50"))
ELEVENLABS_SIMILARITY_BOOST = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))

# FastAPI app initialization
app = FastAPI(
    title="Voice-to-URA Pipeline",
    description="A pipeline for transforming audio into URA-ready voice prompts using Whisper, Local LLM, and ElevenLabs",
    version="1.0.0"
)

# Add CORS middleware
allowed_origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://0.0.0.0:8080",
    "https://amil.lucianaferreira.pro",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files but don't let it intercept API endpoints
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Whisper model once
try:
    whisper_model = WhisperModel(WHISPER_MODEL, device="auto")
    logger.info(f"Loaded Whisper model: {WHISPER_MODEL}")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    whisper_model = None

# Load Mistral model once
try:
    mistral_model = Llama(
        model_path=MISTRAL_MODEL_PATH,
        n_ctx=MISTRAL_N_CTX,
        n_threads=max(1, os.cpu_count() // 2),  # Use half of available cores
        n_batch=512,
        verbose=False
    )
    logger.info(f"Loaded Mistral model: {MISTRAL_MODEL_PATH} with context size {MISTRAL_N_CTX}")
except Exception as e:
    logger.error(f"Failed to load Mistral model: {str(e)}")
    mistral_model = None

# --- 1. TRANSCRIPTION PIPELINE ----------------------------------------------

async def transcribe_audio(path: str) -> str:
    """Transcribe audio using Whisper model"""
    if not whisper_model:
        raise HTTPException(500, "Whisper model failed to load")
    try:
        segments, _ = whisper_model.transcribe(path)
        return "".join(s.text for s in segments).strip()
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")

# --- 2. LLM PIPELINE -------------------------------------------------------

async def check_mistral_health():
    """Check if Mistral model is loaded and ready"""
    if mistral_model is None:
        logger.warning("Mistral model is not loaded")
        return False
    return True

async def ensure_mistral_ready():
    """Ensure Mistral model is loaded and ready"""
    if await check_mistral_health():
        return True
    
    # Try to load the model if not already loaded
    for attempt in range(MAX_RETRIES):
        try:
            global mistral_model
            if mistral_model is None:
                mistral_model = Llama(
                    model_path=MISTRAL_MODEL_PATH,
                    n_ctx=MISTRAL_N_CTX,
                    n_threads=max(1, os.cpu_count() // 2),
                    n_batch=512,
                    verbose=False
                )
                logger.info(f"Loaded Mistral model on attempt {attempt+1} with context size {MISTRAL_N_CTX}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load Mistral model on attempt {attempt+1}: {str(e)}")
            delay = min(RETRY_DELAY * (1.5 ** attempt), 3.0)
            await asyncio.sleep(delay)
    
    logger.error(f"Failed to load Mistral model after {MAX_RETRIES} attempts")
    return False

async def rewrite_to_ura(text: str) -> str:
    """Use Mistral to rewrite text into URA-style format with SSML markup"""
    # Ensure Mistral model is ready
    if not await ensure_mistral_ready():
        raise HTTPException(503, "Language model not available. Please try again in a moment.")
    
    # System prompt for URA rewriting with SSML markup
    system_prompt = MISTRAL_SYSTEM_PROMPT
    
    # User prompt for rewriting
    user_prompt = f'Reescreva o trecho abaixo conforme as **Regras de formatação e marcações SSML**:\n\n"{text}"'
    
    try:
        # Create the full prompt in chat format that Mistral understands
        prompt = f"<s>[INST] {system_prompt} [/INST]</s>[INST] {user_prompt} [/INST]"
        
        # Generate response with Mistral
        logger.info(f"Generating URA rewrite with Mistral (context size: {MISTRAL_N_CTX})")
        response = mistral_model.create_completion(
            prompt=prompt,
            max_tokens=min(2048, MISTRAL_N_CTX // 2),  # Use up to half of context for output
            temperature=0.1,
            top_p=0.9,
            stop=["</s>", "[INST]"],
            echo=False
        )
        
        # Extract the generated text
        if response and "choices" in response and len(response["choices"]) > 0:
            generated_text = response["choices"][0]["text"].strip()
            logger.info("Successfully got URA rewrite from Mistral model")
            return generated_text
        else:
            logger.warning("Received empty or invalid response from Mistral model")
            raise HTTPException(500, "The language model returned an empty response. Please try again.")
    except Exception as e:
        logger.error(f"Error generating with Mistral: {str(e)}")
        raise HTTPException(500, f"Language model error: {str(e)}")

# --- 3. ELEVENLABS TTS INTEGRATION -----------------------------------------

def synthesize_with_elevenlabs(text: str) -> bytes:
    """Returns binary MP3 audio generated by the ElevenLabs API."""
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}'
    headers = {
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg',
    }
    payload = {
        'text': text,
        'model_id': ELEVENLABS_MODEL_ID,
        'voice_settings': {'stability': ELEVENLABS_STABILITY, 'similarity_boost': ELEVENLABS_SIMILARITY_BOOST}
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.RequestException as e:
        logger.error(f"ElevenLabs API error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
        raise HTTPException(500, f"TTS synthesis failed: {str(e)}")

# --- 4. URA TEXT PROCESSING ------------------------------------------------

def validate_ura_text(text: str) -> None:
    """
    Raises HTTPException if text is not URA-compatible.
    Checks:
      - Only contains letters, numbers, basic punctuation (.,;?!)
      - No segment longer than MAX_WORDS words
      - Total length under MAX_CHARS
    """
    MAX_CHARS = 500
    MAX_WORDS = 12

    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"Text too long for a single prompt ({len(text)} chars)")

    # whitelist check
    if re.search(r"[^0-9A-Za-zÀ-ÿ ,\.;\?!\u00C1-\u017F]", text):
        raise HTTPException(400, "Text contains unsupported characters")

    # segment length check
    segments = re.split(r"[\.;\?!]\s*", text)
    for seg in segments:
        word_count = len(seg.strip().split())
        if word_count > MAX_WORDS:
            raise HTTPException(400, f"Segment too long ({word_count} words): \"{seg}\"")

def clean_text(text: str) -> str:
    """
    Normalizes whitespace and punctuation:
      - Collapse multiple spaces
      - Ensure single space after punctuation
      - Strip leading/trailing whitespace
    """
    # collapse whitespace
    t = re.sub(r"\s+", " ", text)
    # ensure space after punctuation
    t = re.sub(r"([\,\.\;\?\!])([^\s])", r"\1 \2", t)
    return t.strip()

def annotate_ura_ssml(text: str, pause_ms: int = 350) -> str:
    """
    Splits on sentence punctuation and re‑joins with SSML <break> tags.
    Returns a full SSML document ready for ElevenLabs.
    """
    # split at end-of-sentence punctuation, keep the punctuation
    parts = re.findall(r".+?[\.;\?!](?:\s|$)", text)
    ssml_chunks = []
    for part in parts:
        clean_part = part.strip()
        # escape XML-sensitive chars if needed
        clean_part = clean_part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ssml_chunks.append(f"{clean_part}<break time=\"{pause_ms}ms\"/>")
    # wrap in <speak> if your TTS supports SSML
    return "<speak>" + " ".join(ssml_chunks) + "</speak>"

# --- 5. API ENDPOINTS ------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api")
async def api_info():
    return {"status": "ready", "message": "Voice-to-URA Pipeline API"}

@app.get("/health")
async def health_check():
    # Check Mistral health
    try:
        mistral_status = await check_mistral_health()
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        mistral_status = False
    
    # Check Whisper model status
    whisper_status = whisper_model is not None
    whisper_detail = "loaded" if whisper_status else "not loaded"
    
    # Check ElevenLabs API key
    elevenlabs_status = ELEVENLABS_API_KEY is not None
    
    # Compute overall status
    overall_status = "ok" if mistral_status and whisper_status and elevenlabs_status else "degraded"
    if not (mistral_status and whisper_status):
        overall_status = "critical"
    
    # Return detailed health information
    return {
        "status": overall_status,
        "version": "1.0.0",
        "timestamp": int(time.time()),
        "components": {
            "mistral": {
                "status": "ready" if mistral_status else "unavailable", 
                "model_path": MISTRAL_MODEL_PATH,
                "context_size": MISTRAL_N_CTX
            },
            "whisper": {
                "status": whisper_detail,
                "model": WHISPER_MODEL
            },
            "elevenlabs": {
                "status": "configured" if elevenlabs_status else "missing API key",
                "voice_id": ELEVENLABS_VOICE_ID,
                "model": ELEVENLABS_MODEL_ID
            }
        }
    }

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Complete URA pipeline:
    1. Transcribe audio with Whisper
    2. Rewrite to URA format with local LLM
    3. Validate, clean and annotate with SSML
    4. Synthesize using ElevenLabs
    5. Return processing results
    """
    # Create a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Set up temporary file paths
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, f"input_{request_id}")
    wav_path = os.path.join(temp_dir, f"input_{request_id}.wav")
    output_path = os.path.join(temp_dir, f"output_{request_id}.mp3")
    
    try:
        # 1. Save uploaded audio
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Convert to WAV for Whisper
        ff = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", wav_path],
            capture_output=True
        )
        if ff.returncode != 0:
            error_msg = ff.stderr.decode().strip()
            logger.error(f"ffmpeg error: {error_msg}")
            raise HTTPException(500, f"Audio conversion error: {error_msg}")
        
        # 2. Transcribe the audio
        transcript = await transcribe_audio(wav_path)
        logger.info(f"Transcribed text: {transcript}")
        
        if not transcript or transcript.strip() == "":
            logger.warning("Empty transcript from Whisper")
            raise HTTPException(400, "Could not understand audio. Please try again.")
        
        # 3. Rewrite via local LLM to URA style
        ura_raw = await rewrite_to_ura(transcript)
        
        # 4. Validate + Clean + Annotate
        validate_ura_text(ura_raw)
        ura_clean = clean_text(ura_raw)
        ura_ssml = annotate_ura_ssml(ura_clean, pause_ms=400)
        
        # 5. Synthesize with ElevenLabs
        audio_bytes = synthesize_with_elevenlabs(ura_ssml)
        
        # 6. Save output file
        with open(output_path, "wb") as out:
            out.write(audio_bytes)
        
        # 7. Return results
        return JSONResponse({
            "status": "ok",
            "request_id": request_id,
            "transcript": transcript,
            "ura_text": ura_clean,
            "ssml": ura_ssml,
            "output_file": output_path
        })
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_audio: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")
    finally:
        # Don't clean up output file as it might be needed, but clean original files
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")

@app.post("/regenerate-audio")
async def regenerate_audio(text: str = Form(...)):
    """
    Regenerate URA audio with edited text:
    1. Validate the edited text
    2. Clean and annotate with SSML
    3. Synthesize using ElevenLabs
    4. Return processing results
    """
    # Create a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Set up temporary file paths
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"output_{request_id}.mp3")
    
    try:
        # 1. Validate the edited text
        validate_ura_text(text)
        
        # 2. Clean and annotate with SSML
        ura_clean = clean_text(text)
        ura_ssml = annotate_ura_ssml(ura_clean, pause_ms=400)
        
        # 3. Synthesize with ElevenLabs
        audio_bytes = synthesize_with_elevenlabs(ura_ssml)
        
        # 4. Save output file
        with open(output_path, "wb") as out:
            out.write(audio_bytes)
        
        # 5. Return results
        return JSONResponse({
            "status": "ok",
            "request_id": request_id,
            "ura_text": ura_clean,
            "ssml": ura_ssml,
            "output_file": output_path
        })
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in regenerate_audio: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")

# --- OPTIONAL: FILE DOWNLOAD ENDPOINT --------------------------------------

@app.get("/download/{request_id}")
async def download_audio(request_id: str):
    """Download the generated audio file by request ID"""
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"output_{request_id}.mp3")
    
    if not os.path.exists(output_path):
        raise HTTPException(404, "Audio file not found")
    
    return FileResponse(
        output_path,
        media_type="audio/mpeg",
        filename=f"ura_audio_{request_id}.mp3"
    ) 