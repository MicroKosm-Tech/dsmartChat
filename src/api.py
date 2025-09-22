import tempfile
import os
import time
import logging
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import base64
from .agent import SimpleMedicalAgent
from .utils import CustomCallBackHandler, thread_local, setup_improved_logging, log_thread_local_state
from src.AESEncryptor import AESEncryptor
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import re

# Import our essential tools and functionality
from .agent_tools import analyze_symptoms, dynamic_doctor_search, store_patient_details
from .query_builder_agent import unified_doctor_search, unified_doctor_search_tool, extract_search_criteria_tool
from .specialty_matcher import SpecialtyDataCache, get_recommended_specialty

# Set up improved logging at startup
setup_improved_logging()

# Configure comprehensive logging for FastAPI and uvicorn
import logging
from src.agent import setup_detailed_logging

# Setup comprehensive logging that captures everything
logger = setup_detailed_logging()

# Configure FastAPI logging
logging.getLogger("fastapi").setLevel(logging.INFO)
logging.getLogger("fastapi").propagate = True

# Configure uvicorn logging
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn").propagate = True
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").propagate = True
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.error").propagate = True

# Preload specialty data cache at application startup
logger = logging.getLogger(__name__)
logger.info("Preloading specialty data cache at application startup")
try:
    specialty_data = SpecialtyDataCache.get_instance()

    # Print loaded subspecialties to console
    unique_subspecialties = set()
    for entry in specialty_data:
        if entry.get("subspecialty"):
            unique_subspecialties.add(entry.get("subspecialty"))

    logger.info(f"Specialty data cache preloaded with {len(specialty_data)} specialty records")
    logger.info(f"Loaded {len(unique_subspecialties)} unique subspecialties:")
    for subspecialty in sorted(unique_subspecialties):
        logger.info(f"  - {subspecialty}")

    # No longer need to log variant mappings as we're using GPT for this now
    logger.info(f"Using GPT for subspecialty matching instead of hardcoded mappings")

except Exception as e:
    logger.error(f"Error preloading specialty data: {str(e)}")
    logger.warning("Application will continue but specialty matching may be affected")

# Workflow notes:
# 1. User describes symptoms → analyze_symptoms → detect symptoms and matching specialties
# 2. Results stored in session history including specialties and subspecialties
# 3. When searching for doctors, structured specialty/subspecialty information is passed 
#    directly to the query builder rather than just using natural language
# 4. This ensures consistent specialty mapping between symptom analysis and doctor search

# Language detection and voice mapping
def detect_language_from_text(text: str) -> str:
    """
    Simple language detection based on character sets and common patterns
    """
    # Check for Arabic characters
    if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text):
        return "ar"
    # Check for Japanese characters (hiragana and katakana) - check this first before Chinese
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return "ja"
    # Check for Korean characters
    elif re.search(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]', text):
        return "ko"
    # Check for Chinese characters (simplified and traditional)
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return "zh"
    # Check for Hindi/Devanagari
    elif re.search(r'[\u0900-\u097F]', text):
        return "hi"
    # Check for Thai
    elif re.search(r'[\u0E00-\u0E7F]', text):
        return "th"
    # Check for Russian
    elif re.search(r'[\u0400-\u04FF]', text):
        return "ru"
    # Check for French (common words) - check before Spanish to avoid conflicts
    elif re.search(r'\b(le|la|les|est|sont|je|tu|il|elle|nous|vous|ils|elles|bonjour|merci|s\'il vous plaît|aujourd\'hui|comment|allez|avez|mal|tête|fièvre)\b', text, re.IGNORECASE):
        return "fr"
    # Check for Spanish (common words)
    elif re.search(r'\b(el|la|los|las|es|son|está|están|tengo|tienes|tiene|tienen|hola|gracias|por favor|hoy|cómo|estás|dolor|cabeza|fiebre)\b', text, re.IGNORECASE):
        return "es"
    # Check for German (common words)
    elif re.search(r'\b(der|die|das|ist|sind|ich|du|er|sie|wir|ihr|sie|hallo|danke|bitte|heute|wie|geht|es|ihnen|kopfschmerzen|fieber)\b', text, re.IGNORECASE):
        return "de"
    # Default to English
    else:
        return "en"

# Voice mapping for different languages
VOICE_MAPPING = {
    "en": {
        "onyx": "echo",
    },
    "ar": {
        "onyx": "echo",
    },
    "es": {
        "onyx": "echo",
    },
    "fr": {
        "onyx": "echo",
    },
    "de": {
        "onyx": "echo",
    },
    "zh": {
        "onyx": "echo",
    },
    "ja": {
        "onyx": "echo",
    },
    "ko": {
        "onyx": "echo",
    },
    "hi": {
        "onyx": "echo",
    },
    "th": {
        "onyx": "echo",
    },
    "ru": {
        "onyx": "echo",
    }
}

def get_voice_for_language(language: str, voice_preference: str = "onyx") -> str:
    """
    Get appropriate voice for the detected language - now always returns 'onyx' for consistency
    """
    return "onyx"

# Custom JSON encoder for Arabic text
class ArabicJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        return super().encode(obj)

def arabic_json_dumps(obj, **kwargs):
    return json.dumps(obj, cls=ArabicJSONEncoder, ensure_ascii=False, **kwargs)

class ArabicJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return arabic_json_dumps(content).encode("utf-8")

# Create FastAPI app with our custom JSON response class as default
app = FastAPI(default_response_class=ArabicJSONResponse)
engine = SimpleMedicalAgent()
callBackHandler = CustomCallBackHandler()
encryptor = AESEncryptor(os.getenv("ENCRYPTION_SALT", "default_salt"))

# Set environment variables - load from .env directly
load_dotenv()  # This will load variables from .env file

# Get API key from environment with better error handling
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not api_key:
    error_msg = """
    ⚠️ ERROR: No OpenAI API key found in environment variables.
    
    To fix this:
    1. Create a .env file in the project root if it doesn't exist
    2. Add your OpenAI API key: OPENAI_API_KEY=your_key_here
    3. Restart the application
    
    Alternatively, you can set the API_KEY or OPENAI_API_KEY environment variable directly.
    """
    logger.error(error_msg)
else:
    logger.info("✅ OpenAI API key found in environment variables")

# Set environment variables
os.environ["OPENAI_API_KEY"] = api_key if api_key else ""

# Define allowed origins from environment variable or use defaults
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:8509").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from openai import OpenAI
# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=api_key)

def get_token_usage(response):
    # Extract token usage from the API response
    token_usage = response['usage']
    tokens_used = token_usage['total_tokens']
    return tokens_used

@app.post("/chat")
async def chat(
    message: str = Form(None),
    session_id: str = Form(...),
    audio: UploadFile = File(None),
    lat : Optional[float] = Form(None),
    long : Optional[float] = Form(None),
):
    start_time = time.time()
    detected_language = "en"  # default language
    
    try:

        if session_id == None or session_id == "":
            return {
                "response": {
                    "error" : "EMPTY SESSION ID NOT ALLOWED"
                }
            }

        
        pattern = r'^www\|.+'

        if re.match(pattern, session_id):
            print("Valid Session ID match: ", session_id)
        else:
            return {
                "response": {
                    "error" : "INVALID SESSION ID FORMAT"
                }
            }
        # Handle audio input
        if audio:
            if not audio.filename or not audio.filename.lower().endswith(".wav"):
                raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are allowed.")
            
                # Process audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file_path = temp_file.name
                contents = await audio.read()
                temp_file.write(contents)
            
            try:
                    # Transcribe audio with language detection
                with open(temp_file_path, "rb") as audio_file:
                    transcription_response = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file,
                        response_format="verbose_json"
                    )
                    
                    transcription = transcription_response.text
                    # Use Whisper's language detection if available, otherwise detect from text
                    if hasattr(transcription_response, 'language'):
                        detected_language = transcription_response.language
                    else:
                        detected_language = detect_language_from_text(transcription)
                    
                    print("-------------------------")
                    print("LAT: ", lat)
                    print("LONG: ", long)
                    print("DETECTED LANGUAGE: ", detected_language)
                    print("-------------------------")
                    
                    # Process transcribed message
                    response = process_message(transcription, session_id, lat, long)
                    
                    # Generate audio response with detected language
                    audio_base64 = await text_to_speech_with_language(
                        response['response']['message'], 
                        detected_language
                    )
                    
                    # Return a single response object with audio data added
                    if "response" in response:
                        # Add audio data to existing response
                        response["response"]["Audio_base_64"] = audio_base64
                        response["response"]["transcription"] = transcription
                        response["response"]["detected_language"] = detected_language
                        return response
                    else:
                        # Create a properly formatted response
                        return {
                            "response": {
                                "message": response.get('message', ''),
                                "transcription": transcription,
                                "Audio_base_64": audio_base64,
                                "detected_language": detected_language,
                                "patient": response.get('patient', {"session_id": session_id}),
                                "data": response.get('data', [])
                            }
                        }
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
            # Handle text input
        if message:
                # Detect language from text input
                detected_language = detect_language_from_text(message)
                
                print("-------------------------")
                print("LAT: ", lat)
                print("LONG: ", long)
                print("DETECTED LANGUAGE: ", detected_language)
                print("-------------------------")
                
                response = process_message(message, session_id, lat, long)
                return response
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {
            "response": {
                "message": f"I encountered an error processing your request: {str(e)}",
                "patient": {"session_id": session_id},
                "data": []
            }
        }

def process_message(message: str, session_id: str, lat: Optional[float] = None, long: Optional[float] = None) -> dict:
    """Process incoming message by sending it to the agent"""
    try:
        logger.info(f"Processing message for session {session_id}")
        if lat is not None and long is not None:
            logger.info(f"Using coordinates: lat={lat}, long={long}")
        
        # Forward the message to the agent and get response
        response = engine.process_message(message, session_id, lat, long)
        
        # Return the response without wrapping it in another response object
        return response

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        # Main agent handles error responses - no additional AI calls
        error_message = "I apologize, but I encountered an error while processing your request. Please try rephrasing your question or contact support if the issue persists."
        
        return {
            "response": {
                "message": error_message,
                "patient": {"session_id": session_id},
                "data": []
            }
        }

class ImageRequest(BaseModel):
    base64_image: str  # Expecting a base64-encoded image string

@app.post("/analyze-image")
async def analyze_image(request: ImageRequest):
    try:
        # Extract the base64 image from request
        base64_image = request.base64_image

        # Ensure it is a valid base64 string
        try:
            base64.b64decode(base64_image.split(",")[1])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image format")

        # Call OpenAI API
        logger.info("********************************")
        logger.info("CALLING IMAGE ANALYSIS AGENT")
        logger.info("********************************")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is an image of a medical condition. Your job is to provide the corresponding medical specialty along with relevant medical tags, comma-separated."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image,  
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        description = response.choices[0].message.content
        print(description)
        
        return {"status": "success", "description": description}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
    
@app.get("/")
def read_root():
    return {"response": "Hello World"}

@app.post("/reset")
async def reset(session_id: str = Form(...)):
    """Reset a specific session, clearing all history and thread_local data"""
    try:
        logger.info(f"Resetting session {session_id}")
        
        # Reset engine history for this session if it exists
        from .agent import store
        if session_id in store:
            logger.info(f"Clearing chat history for session {session_id}")
            store[session_id].clear()
            del store[session_id]
        
        # Reset thread_local data if it matches this session
        if hasattr(thread_local, 'session_id') and thread_local.session_id == session_id:
            logger.info(f"Clearing thread_local data for session {session_id}")
            # Log current state before clearing
            log_thread_local_state(logger, session_id)
            # Clear all session-specific attributes
            for attr in dir(thread_local):
                if not attr.startswith('_'):
                    logger.info(f"Deleting thread_local.{attr}")
                    delattr(thread_local, attr)
        
        return {"status": "success", "message": f"Session {session_id} has been reset"}
    except Exception as e:
        logger.error(f"Error resetting session {session_id}: {str(e)}")
        return {"status": "error", "message": str(e)}
        
async def text_to_speech_with_language(text: str, language: str = "en", voice_preference: str = "onyx") -> str:
    """
    Generate speech using OpenAI TTS with language-appropriate voice
    """
    try:
        # Get appropriate voice for the language
        voice = get_voice_for_language(language, voice_preference)
        
        logger.info(f"Generating TTS for language: {language}, using voice: {voice}")
        
        # Add system instructions for Saudi Arabic speaker with formal tone
        system_instructions = "You are a Saudi Arabic speaker, but your tone must be formal and your pitch and speed must be audible."
        
        # Enhanced prompt with system instructions
        enhanced_text = f"[System: {system_instructions}] {text}"
        
        # Use OpenAI TTS with proper type casting
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",  # You can use "tts-1-hd" for higher quality
            voice=voice,  # type: ignore
            input=enhanced_text
        )
        
        # Convert to base64
        audio_data = response.content
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        logger.info(f"TTS generation successful for language: {language}")
        return audio_base64
        
    except Exception as e:
        logger.error(f"Error in OpenAI TTS: {str(e)}")
        # Fallback to default voice
        return await text_to_speech_fallback(text)

async def text_to_speech_fallback(text: str) -> str:
    """
    Fallback TTS when OpenAI TTS fails
    """
    try:
        logger.warning("Using fallback TTS method")
        
        # Add system instructions for fallback as well
        system_instructions = "You are a Saudi Arabic speaker, but your tone must be formal and your pitch and speed must be audible."
        enhanced_text = f"[System: {system_instructions}] {text}"
        
        # Try with default voice
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="onyx",  # type: ignore
            input=enhanced_text
        )
        
        audio_data = response.content
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        return audio_base64
        
    except Exception as e:
        logger.error(f"Fallback TTS also failed: {str(e)}")
        # Return empty audio if all fails
        return ""

# Keep the old function for backward compatibility
async def text_to_speech(text: str) -> str:
    """
    Legacy function - now calls the new multi-language TTS
    """
    return await text_to_speech_with_language(text, "en", "onyx")


