from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import torch
import whisper
import shutil
import logging
from typing import Optional, Dict, Any, List
import uvicorn
import json
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whisper-flamingo-api")

app = FastAPI(
    title="Whisper-Flamingo API",
    description="API for audio-visual speech recognition using Whisper-Flamingo",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to store model instances
whisper_model = None
tokenizer = None
model_type = "medium"  # Default model type
device = "cuda" if torch.cuda.is_available() else "cpu"
use_av_hubert_encoder = 1
av_fusion = "separate"
whisper_path = "models/"
av_hubert_path = "av_hubert/avhubert/"
av_hubert_ckpt = "models/large_noise_pt_noise_ft_433h_only_weights.pt"  # Path to AV-HuBERT weights

# Multilingual support
supported_languages = {
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "it": "Italian", 
    "fr": "French",
    "pt": "Portuguese",
    "ru": "Russian",
    "lrs2": "LRS2 (English)"
}

class TranscriptionRequest(BaseModel):
    language: str = "en"
    noise_snr: int = 1000  # Default to clean audio (no noise)
    task: str = "transcribe"  # Options: transcribe, En-X, X-En
    modalities: str = "avsr"  # Options: avsr (audio-visual), asr (audio-only), vsr (video-only)
    beam_size: int = 1

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    metrics: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up Whisper-Flamingo API using device: {device}")
    # Model will be loaded on demand to save resources

def load_model(language="en", modalities="avsr", checkpoint_path=None):
    """Load the appropriate Whisper-Flamingo model based on parameters"""
    global whisper_model, tokenizer
    
    if whisper_model is not None:
        # If model is already loaded, return it
        return whisper_model, tokenizer
    
    logger.info(f"Loading Whisper model: {model_type}")
    
    # Determine if using multilingual tokenizer
    multilingual = True if 'large' in model_type or 'en' not in model_type else False
    logger.info(f"Using multilingual tokenizer: {multilingual}")
    
    # Determine task (transcribe or translate)
    task = 'translate' if language == 'X-En' else 'transcribe'
    
    # Create tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=multilingual, task=task)
    
    # Load Whisper model with appropriate settings
    whisper_model = whisper.load_model(
        model_type, 
        download_root=whisper_path,
        video=True if modalities != "asr" else False,
        video_model_path=av_hubert_ckpt if modalities != "asr" else None,
        av_hubert_path=av_hubert_path if modalities != "asr" else None,
        av_hubert_encoder=use_av_hubert_encoder if modalities != "asr" else 0,
        av_fusion=av_fusion if modalities != "asr" else "None",
        add_gated_x_attn=1 if av_fusion == 'separate' and modalities != "asr" else 0
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                # Remove 'model.' prefix from keys
                state_dict_updated = {k[6:]: v for k, v in state_dict.items()}
                try:
                    whisper_model.load_state_dict(state_dict_updated)
                except Exception as e:
                    logger.warning(f"Loading with strict=False due to: {str(e)}")
                    whisper_model.load_state_dict(state_dict_updated, strict=False)
            else:
                try:
                    whisper_model.load_state_dict(state_dict)
                except Exception as e:
                    logger.warning(f"Loading with strict=False due to: {str(e)}")
                    whisper_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model checkpoint: {str(e)}")
    
    # Move model to appropriate device
    whisper_model.to(device)
    whisper_model.eval()
    
    return whisper_model, tokenizer

def process_media(audio_file_path, video_file_path, language, noise_snr, task, modalities, beam_size):
    """Process audio/video and return transcription"""
    # Determine the appropriate checkpoint path based on language and modalities
    checkpoint_path = get_checkpoint_path(language, modalities)
    
    # Load model
    model, tokenizer = load_model(language, modalities, checkpoint_path)
    
    # Set up decoding options
    task_type = 'translate' if task == 'X-En' else 'transcribe'
    options = whisper.DecodingOptions(
        task=task_type, 
        language=language.replace('lrs2', 'en'),  # Handle LRS2 special case
        fp16=(device == "cuda"),
        without_timestamps=True,
        beam_size=None if beam_size == 1 else beam_size,
    )
    
    # TODO: Implement the actual media processing code
    # This will depend on how your whisper_decode_video.py script processes inputs
    
    # Mock implementation - replace with actual implementation
    result = {"text": "This is a mock transcription. Implement actual processing."}
    
    return result

def get_checkpoint_path(language, modalities):
    """
    Determine the appropriate model checkpoint based on language and modality
    """
    # This is a simplified implementation - you'll need to expand this based on your models
    if modalities == "avsr":
        if language == "en":
            return "models/whisper-flamingo_en-x_medium.pt"
        elif language == "lrs2":
            return "models/whisper-flamingo_lrs2_medium.pt"
        else:
            # For other languages, use multilingual model
            return "models/whisper-flamingo_multi-all_medium.pt"
    else:  # ASR mode
        if language == "en":
            return "models/whisper_en-x_medium.pt"
        elif language == "lrs2":
            return "models/whisper_lrs2_medium.pt"
        else:
            # For other languages, use multilingual model
            return "models/whisper_multi-all_medium.pt"

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_file(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
    params: TranscriptionRequest = Form(...)
):
    """
    Transcribe audio/video file using Whisper-Flamingo
    """
    # Validate language
    if params.language not in supported_languages and params.language != "auto":
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {params.language}. Supported languages: {list(supported_languages.keys())}"
        )
    
    # Validate modality and files
    if params.modalities == "avsr" and not video_file:
        raise HTTPException(
            status_code=400,
            detail="Video file is required for audio-visual speech recognition mode"
        )
    
    if params.modalities in ["avsr", "asr"] and not audio_file:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required for modes that use audio"
        )
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    audio_path = None
    video_path = None
    
    try:
        # Save uploaded files
        if audio_file:
            audio_path = os.path.join(temp_dir, audio_file.filename)
            with open(audio_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
        
        if video_file:
            video_path = os.path.join(temp_dir, video_file.filename)
            with open(video_path, "wb") as f:
                content = await video_file.read()
                f.write(content)
        
        # Process the files
        result = process_media(
            audio_path,
            video_path,
            params.language,
            params.noise_snr,
            params.task,
            params.modalities,
            params.beam_size
        )
        
        # Add cleanup task
        background_tasks.add_task(cleanup_temp_files, temp_dir)
        
        return TranscriptionResponse(
            text=result["text"],
            language=params.language,
            metrics={"confidence": 0.95}  # Mock confidence score
        )
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_temp_files(temp_dir):
    """Clean up temporary files"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

@app.get("/languages/")
async def get_languages():
    """Get list of supported languages"""
    return {"languages": supported_languages}

@app.get("/models/")
async def get_models():
    """Get information about available models"""
    # This is a simplified implementation - you'll need to expand this based on your models
    models = {
        "whisper_en-x_medium": {"type": "audio", "languages": ["en", "es", "fr", "it", "pt", "ru", "el"]},
        "whisper-flamingo_en-x_medium": {"type": "audio-visual", "languages": ["en", "es", "fr", "it", "pt", "ru", "el"]},
        "whisper_lrs2_medium": {"type": "audio", "languages": ["en"]},
        "whisper-flamingo_lrs2_medium": {"type": "audio-visual", "languages": ["en"]},
        "whisper_multi-all_medium": {"type": "audio", "languages": ["en", "ar", "de", "el", "es", "it", "fr", "pt", "ru"]},
        "whisper-flamingo_multi-all_medium": {"type": "audio-visual", "languages": ["en", "ar", "de", "el", "es", "it", "fr", "pt", "ru"]}
    }
    return {"models": models}

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "device": device}

if __name__ == "__main__":
    uvicorn.run("whisper_api:app", host="0.0.0.0", port=8000, reload=True)