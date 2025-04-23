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
from xml.etree.ElementTree import fromstring, ParseError

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
        # Create the prompt in chat format that Mistral understands, avoiding duplicate <s> tags
        prompt = f"[INST] {system_prompt} [/INST][INST] {user_prompt} [/INST]"
        
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
    """
    Sintetiza texto em áudio usando a API ElevenLabs.
    Suporta texto comum ou marcado com SSML.
    Retorna o áudio em formato MP3.
    """
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}'
    
    # Verifica se a API key existe
    if not ELEVENLABS_API_KEY:
        logger.error("❌ API key do ElevenLabs não configurada")
        raise HTTPException(500, "ElevenLabs API key não configurada")
    
    headers = {
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg',
    }
    
    # Verificar se o texto tem marcação SSML
    is_ssml = text.strip().startswith("<speak>")
    
    if is_ssml:
        logger.info("🔊 Enviando texto com formatação SSML para ElevenLabs")
    else:
        logger.info("🔊 Enviando texto plano para ElevenLabs")
    
    payload = {
        'text': text,
        'model_id': ELEVENLABS_MODEL_ID,
        'voice_settings': {'stability': ELEVENLABS_STABILITY, 'similarity_boost': ELEVENLABS_SIMILARITY_BOOST}
    }
    
    # Se for SSML, adicionar config de markup
    if is_ssml:
        payload['text_markup_type'] = 'ssml'
    
    try:
        logger.info(f"🔊 Requisição para ElevenLabs: voice={ELEVENLABS_VOICE_ID}, model={ELEVENLABS_MODEL_ID}")
        resp = requests.post(url, headers=headers, json=payload)
        
        if resp.status_code == 400:
            # Tentar extrair detalhes do erro
            try:
                error_detail = resp.json().get('detail', {})
                error_message = error_detail.get('message', 'Erro desconhecido')
                logger.error(f"❌ ElevenLabs rejeitou a requisição: {error_message}")
                
                # Fallback: Se falhou com SSML, tentar remover tags
                if is_ssml and "SSML" in error_message:
                    logger.warning("⚠️ Tentando fallback para texto sem SSML")
                    # Extrair texto puro do SSML
                    plain_text = re.sub(r'<[^>]+>', '', text)
                    plain_text = plain_text.replace('</speak>', '').replace('<speak>', '')
                    
                    # Construir novo payload
                    fallback_payload = {
                        'text': plain_text,
                        'model_id': ELEVENLABS_MODEL_ID,
                        'voice_settings': {'stability': ELEVENLABS_STABILITY, 'similarity_boost': ELEVENLABS_SIMILARITY_BOOST}
                    }
                    
                    logger.info(f"🔄 Tentando novamente com texto plano: {len(plain_text)} caracteres")
                    resp = requests.post(url, headers=headers, json=fallback_payload)
                    
                    if not resp.ok:
                        logger.error(f"❌ Fallback também falhou: {resp.status_code}")
                        raise HTTPException(500, f"Falha na síntese de voz: {error_message}")
                else:
                    raise HTTPException(500, f"Falha na síntese de voz: {error_message}")
            except (ValueError, KeyError) as e:
                logger.error(f"❌ Erro ao processar resposta de erro do ElevenLabs: {str(e)}")
                raise HTTPException(500, f"Falha na síntese de voz (código {resp.status_code})")
        
        if not resp.ok:
            logger.error(f"❌ Erro na API ElevenLabs: {resp.status_code}")
            raise HTTPException(500, f"Falha na síntese de voz (código {resp.status_code})")
        
        audio_content = resp.content
        logger.info(f"✅ Áudio sintetizado com sucesso: {len(audio_content)} bytes")
        return audio_content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erro de rede ao acessar ElevenLabs: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            status = e.response.status_code if hasattr(e.response, 'status_code') else 'Desconhecido'
            logger.error(f"Resposta: Código {status}")
            logger.error(f"Conteúdo: {e.response.text if hasattr(e.response, 'text') else 'Não disponível'}")
        raise HTTPException(500, f"Falha na conexão com serviço de síntese de voz: {str(e)}")
    except Exception as e:
        logger.exception(f"❌ Erro inesperado na síntese de voz: {str(e)}")
        raise HTTPException(500, f"Falha na síntese de voz: {str(e)}")

# --- 4. URA TEXT PROCESSING ------------------------------------------------

def validate_ura_text(text: str) -> None:
    """
    Valida o texto URA, diferenciando entre SSML e texto plano.
    Para SSML: verifica se é XML bem formado
    Para texto plano: aplica regras mais estritas de caracteres permitidos
    """
    if text is None:
        logger.error("❌ Texto URA é None")
        raise HTTPException(400, "Texto URA está vazio")
        
    if not isinstance(text, str):
        logger.error(f"❌ Texto URA não é string: {type(text)}")
        raise HTTPException(400, f"Tipo de texto inválido: {type(text)}")
    
    # Verifica se o texto está vazio
    if not text.strip():
        logger.error("❌ Texto URA está vazio após strip()")
        raise HTTPException(400, "Texto URA está vazio")

    # Limite de caracteres (para ambos tipos)
    MAX_CHARS = 5000  # Aumentado para acomodar textos SSML
    if len(text) > MAX_CHARS:
        logger.error(f"❌ Texto URA muito longo: {len(text)} caracteres")
        raise HTTPException(400, f"Texto muito longo para um único prompt ({len(text)} caracteres, máximo {MAX_CHARS})")

    # Detecta se é SSML ou texto plano
    is_ssml = text.strip().startswith("<speak>") or (
        "<speak>" in text and "</speak>" in text
    )
    
    if is_ssml:
        logger.info("🔍 Texto contém marcação SSML, validando como XML")
        # Para SSML: verificar se é XML bem formado
        try:
            # Tenta fazer parse como XML
            xml_text = text
            # Certifica que tem a tag root <speak>
            if not xml_text.strip().startswith("<speak>"):
                xml_text = f"<speak>{xml_text}</speak>"
            
            fromstring(xml_text)
            logger.info("✅ Validação SSML bem-sucedida")
        except ParseError as e:
            logger.error(f"❌ SSML malformado: {str(e)}")
            # Registrar o texto problemático para debug (com truncamento)
            max_log_len = 200
            truncated = text[:max_log_len] + ("..." if len(text) > max_log_len else "")
            logger.error(f"Texto SSML problemático: {truncated}")
            
            # Em produção, podemos tentar fazer um fallback para texto plano
            # Aqui, vamos apenas aceitar mesmo com SSML inválido
            logger.warning("⚠️ Aceitando SSML potencialmente inválido para não interromper o pipeline")
            return
            # Ou descomente a linha abaixo para rejeitar SSML inválido
            # raise HTTPException(400, f"SSML malformado: {str(e)}")
    else:
        logger.info("🔍 Texto sem marcação SSML, validando como texto plano")
        # Para texto plano: validar caracteres permitidos
        MAX_WORDS = 15  # Limite de palavras por segmento para texto plano
        
        # Lista de caracteres permitidos para texto plano
        # Mais restritiva que para SSML
        caracteres_proibidos = set('=><"%/:')
        caracteres_invalidos = [c for c in text if c in caracteres_proibidos]
        
        if caracteres_invalidos:
            caracteres_unicos = set(caracteres_invalidos)
            logger.error(f"❌ Caracteres não permitidos no texto plano: {caracteres_unicos}")
            raise HTTPException(400, f"O texto contém caracteres não suportados: {', '.join(caracteres_unicos)}")
        
        # Verifica comprimento dos segmentos (apenas para texto plano)
        segments = re.split(r"[\.;\?!]\s*", text)
        for seg in segments:
            word_count = len(seg.strip().split())
            if word_count > MAX_WORDS:
                truncated_seg = seg[:50] + "..." if len(seg) > 50 else seg
                logger.error(f"❌ Segmento muito longo: {word_count} palavras - '{truncated_seg}'")
                raise HTTPException(400, f"Segmento muito longo ({word_count} palavras, máximo {MAX_WORDS}): \"{truncated_seg}\"")
    
    logger.info("✅ Validação do texto URA bem-sucedida")

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
    Prepara o texto para síntese de voz com marcações SSML.
    Se o texto já contiver marcações SSML, retorna conforme está.
    Caso contrário, adiciona marcações básicas de pausa entre frases.
    """
    # Se já estiver no formato SSML completo, retornar como está
    if text.strip().startswith("<speak>") and text.strip().endswith("</speak>"):
        logger.info("✅ Texto já contém marcação SSML completa, mantendo como está")
        return text
    
    # Se contém algumas tags SSML mas não a estrutura completa
    if "<" in text and ">" in text and (
        "<speak>" in text or 
        "<break" in text or 
        "<prosody" in text or 
        "<emphasis" in text
    ):
        logger.info("⚠️ Texto contém algumas tags SSML, mas estrutura incompleta")
        # Garantir que tenha tag speak envolvendo o conteúdo
        if not text.strip().startswith("<speak>"):
            text = f"<speak>{text}"
        if not text.strip().endswith("</speak>"):
            text = f"{text}</speak>"
        return text
    
    # Texto plano - aplicar formatação SSML básica
    logger.info("🔍 Aplicando formatação SSML básica ao texto plano")
    
    # Escapar caracteres especiais XML se presentes
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Dividir em frases e adicionar pausas
    parts = re.findall(r".+?[\.;\?!](?:\s|$)", text)
    if not parts:  # Se não encontrou divisões claras, trata como frase única
        parts = [text]
        
    ssml_chunks = []
    for part in parts:
        clean_part = part.strip()
        ssml_chunks.append(f"{clean_part}<break time=\"{pause_ms}ms\"/>")
    
    # Montar SSML final
    result = "<speak>" + " ".join(ssml_chunks) + "</speak>"
    logger.info(f"✅ SSML gerado: {len(result)} caracteres")
    
    return result

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
    logger.info(f"📝 Iniciando processamento de audio [request_id={request_id}]")
    
    # Set up temporary file paths
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, f"input_{request_id}")
    wav_path = os.path.join(temp_dir, f"input_{request_id}.wav")
    output_path = os.path.join(temp_dir, f"output_{request_id}.mp3")
    
    try:
        # 1. Save uploaded audio
        with open(input_path, "wb") as f:
            content = await file.read()
            if not content:
                logger.error("❌ Arquivo de áudio vazio recebido")
                raise HTTPException(400, "Arquivo de áudio vazio. Por favor, envie um áudio válido.")
            f.write(content)
            logger.info(f"✅ Áudio salvo: {input_path} ({len(content)} bytes)")
        
        # Convert to WAV for Whisper
        logger.info(f"🔄 Convertendo para WAV: {wav_path}")
        ff = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", wav_path],
            capture_output=True
        )
        if ff.returncode != 0:
            error_msg = ff.stderr.decode().strip()
            logger.error(f"❌ Erro ffmpeg: {error_msg}")
            raise HTTPException(400, f"Erro na conversão de áudio: {error_msg}")
        
        # 2. Transcribe the audio
        logger.info(f"🎤 Transcrevendo áudio: {wav_path}")
        try:
            transcript = await transcribe_audio(wav_path)
            logger.info(f"✅ Texto transcrito ({len(transcript)} caracteres): {transcript}")
            
            if not transcript or transcript.strip() == "":
                logger.warning("❌ Transcrição vazia do Whisper")
                raise HTTPException(400, "Não foi possível entender o áudio. Por favor, tente novamente com uma gravação mais clara.")
        except Exception as e:
            logger.exception(f"❌ Erro na transcrição: {str(e)}")
            raise HTTPException(500, f"Falha na transcrição: {str(e)}")
        
        # 3. Rewrite via local LLM to URA style
        logger.info(f"🤖 Enviando para reescrita Mistral: {len(transcript)} caracteres")
        try:
            ura_raw = await rewrite_to_ura(transcript)
            if not ura_raw or not isinstance(ura_raw, str) or ura_raw.strip() == "":
                logger.error(f"❌ Resultado inválido do Mistral: {type(ura_raw)}")
                raise HTTPException(500, "O modelo de linguagem retornou uma resposta inválida. Por favor, tente novamente.")
            
            # Truncate if needed to prevent downstream issues
            if len(ura_raw) > 2000:
                logger.warning(f"⚠️ Texto URA muito longo, truncando de {len(ura_raw)} para 2000 caracteres")
                ura_raw = ura_raw[:2000]
                
            logger.info(f"✅ Texto URA gerado ({len(ura_raw)} caracteres): {ura_raw}")
        except Exception as e:
            logger.exception(f"❌ Erro na reescrita URA: {str(e)}")
            raise HTTPException(500, f"Falha na geração do formato URA: {str(e)}")
        
        # 4. Validate + Clean + Annotate
        logger.info("🔍 Validando e formatando texto URA")
        try:
            validate_ura_text(ura_raw)
            ura_clean = clean_text(ura_raw)
            ura_ssml = annotate_ura_ssml(ura_clean, pause_ms=400)
            logger.info(f"✅ SSML gerado ({len(ura_ssml)} caracteres)")
        except HTTPException as he:
            # Pass through HTTP exceptions with their status code
            logger.error(f"❌ Validação falhou: {str(he.detail)}")
            raise he
        except Exception as e:
            logger.exception(f"❌ Erro na formatação: {str(e)}")
            raise HTTPException(500, f"Falha na formatação do texto: {str(e)}")
        
        # 5. Synthesize with ElevenLabs
        logger.info("🔊 Sintetizando áudio com ElevenLabs")
        try:
            audio_bytes = synthesize_with_elevenlabs(ura_ssml)
            if not audio_bytes or len(audio_bytes) < 100:
                logger.error(f"❌ ElevenLabs retornou áudio inválido: {len(audio_bytes) if audio_bytes else 0} bytes")
                raise HTTPException(500, "Falha na síntese de voz. Por favor, tente novamente.")
            
            logger.info(f"✅ Áudio sintetizado: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.exception(f"❌ Erro na síntese: {str(e)}")
            raise HTTPException(500, f"Falha na síntese de voz: {str(e)}")
        
        # 6. Save output file
        try:
            with open(output_path, "wb") as out:
                out.write(audio_bytes)
            logger.info(f"✅ Arquivo de áudio salvo: {output_path}")
        except Exception as e:
            logger.exception(f"❌ Erro ao salvar áudio: {str(e)}")
            raise HTTPException(500, f"Falha ao salvar arquivo de áudio: {str(e)}")
        
        # 7. Prepare and return final response
        logger.info(f"🏁 Montando resposta final para request_id={request_id}")
        try:
            response_data = {
                "status": "ok",
                "request_id": request_id,
                "transcript": transcript,
                "ura_text": ura_clean,
                "ssml": ura_ssml,
                "output_file": output_path
            }
            # Validar explicitamente que todos os campos estão presentes e são do tipo correto
            if not all(k in response_data for k in ["status", "request_id", "transcript", "ura_text"]):
                logger.error(f"❌ Resposta incompleta: {response_data.keys()}")
                raise ValueError("Resposta incompleta")
                
            logger.info("✅ Processamento completo com sucesso")
            return JSONResponse(response_data)
        except Exception as e:
            logger.exception(f"❌ Erro ao montar resposta JSON final: {str(e)}")
            # Resposta simplificada de fallback
            return JSONResponse({
                "status": "partial_success",
                "request_id": request_id,
                "error": f"Resposta parcial devido a erro: {str(e)}",
                "transcript": transcript if "transcript" in locals() else None,
                "output_file": output_path if os.path.exists(output_path) else None
            })
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"❌ Erro inesperado em process_audio: {str(e)}")
        raise HTTPException(500, f"Erro no servidor: {str(e)}")
    finally:
        # Don't clean up output file as it might be needed, but clean original files
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as e:
            logger.warning(f"⚠️ Erro na limpeza: {str(e)}")

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
    logger.info(f"📝 Iniciando regeneração de áudio [request_id={request_id}]")
    
    # Set up temporary file paths
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"output_{request_id}.mp3")
    
    try:
        # Verificar se o texto foi recebido
        if not text:
            logger.error("❌ Texto vazio recebido para regeneração")
            raise HTTPException(400, "Texto vazio. Por favor, forneça um texto para geração de áudio.")
            
        logger.info(f"📄 Texto recebido para regeneração ({len(text)} caracteres)")
        
        # 1. Validate the edited text
        logger.info("🔍 Validando texto URA")
        try:
            validate_ura_text(text)
        except HTTPException as he:
            logger.error(f"❌ Validação falhou: {str(he.detail)}")
            raise he
        except Exception as e:
            logger.exception(f"❌ Erro na validação: {str(e)}")
            raise HTTPException(500, f"Falha na validação do texto: {str(e)}")
        
        # 2. Clean and annotate with SSML
        logger.info("📝 Formatando texto em SSML")
        try:
            ura_clean = clean_text(text)
            ura_ssml = annotate_ura_ssml(ura_clean, pause_ms=400)
            logger.info(f"✅ SSML gerado ({len(ura_ssml)} caracteres)")
        except Exception as e:
            logger.exception(f"❌ Erro na formatação SSML: {str(e)}")
            raise HTTPException(500, f"Falha na formatação SSML: {str(e)}")
        
        # 3. Synthesize with ElevenLabs
        logger.info("🔊 Sintetizando áudio com ElevenLabs")
        try:
            audio_bytes = synthesize_with_elevenlabs(ura_ssml)
            if not audio_bytes or len(audio_bytes) < 100:
                logger.error(f"❌ ElevenLabs retornou áudio inválido: {len(audio_bytes) if audio_bytes else 0} bytes")
                raise HTTPException(500, "Falha na síntese de voz. Por favor, tente novamente.")
            
            logger.info(f"✅ Áudio sintetizado: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.exception(f"❌ Erro na síntese: {str(e)}")
            raise HTTPException(500, f"Falha na síntese de voz: {str(e)}")
        
        # 4. Save output file
        try:
            with open(output_path, "wb") as out:
                out.write(audio_bytes)
            logger.info(f"✅ Arquivo de áudio salvo: {output_path}")
        except Exception as e:
            logger.exception(f"❌ Erro ao salvar áudio: {str(e)}")
            raise HTTPException(500, f"Falha ao salvar arquivo de áudio: {str(e)}")
        
        # 5. Return results
        logger.info(f"🏁 Montando resposta final para request_id={request_id}")
        try:
            response_data = {
                "status": "ok",
                "request_id": request_id,
                "ura_text": ura_clean,
                "ssml": ura_ssml,
                "output_file": output_path
            }
            logger.info("✅ Regeneração completa com sucesso")
            return JSONResponse(response_data)
        except Exception as e:
            logger.exception(f"❌ Erro ao montar resposta JSON final: {str(e)}")
            # Resposta simplificada de fallback
            return JSONResponse({
                "status": "partial_success",
                "request_id": request_id,
                "error": f"Resposta parcial devido a erro: {str(e)}",
                "output_file": output_path if os.path.exists(output_path) else None
            })
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"❌ Erro inesperado em regenerate_audio: {str(e)}")
        raise HTTPException(500, f"Erro no servidor: {str(e)}")

@app.get("/download/{request_id}")
async def download_audio(request_id: str):
    """Download the generated audio file by request ID"""
    logger.info(f"📥 Solicitação de download para request_id={request_id}")
    
    # Validar ID da requisição
    if not request_id or not re.match(r'^[a-f0-9\-]+$', request_id):
        logger.error(f"❌ ID de requisição inválido: {request_id}")
        raise HTTPException(400, "ID de requisição inválido")
    
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"output_{request_id}.mp3")
    
    if not os.path.exists(output_path):
        logger.error(f"❌ Arquivo de áudio não encontrado: {output_path}")
        raise HTTPException(404, "Arquivo de áudio não encontrado. Talvez tenha expirado ou nunca foi gerado.")
    
    try:
        file_size = os.path.getsize(output_path)
        logger.info(f"✅ Enviando arquivo de áudio: {output_path} ({file_size} bytes)")
        
        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename=f"ura_audio_{request_id}.mp3"
        )
    except Exception as e:
        logger.exception(f"❌ Erro ao enviar arquivo: {str(e)}")
        raise HTTPException(500, f"Erro ao enviar arquivo: {str(e)}") 