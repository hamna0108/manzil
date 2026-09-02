from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.schemas.voice import TranscriptionResponse
from app.services.voice_service import (
    MAX_AUDIO_SIZE_BYTES,
    SUPPORTED_MIME_TYPES,
    transcribe_audio,
)

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)):
    # Clean the MIME type by dropping anything after the semicolon
    base_mime_type = audio.content_type.split(";")[0].strip()

    if base_mime_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio format '{audio.content_type}'. "
                f"Supported: {sorted(SUPPORTED_MIME_TYPES)}."
            ),
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio file.")
        
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds the {MAX_AUDIO_SIZE_BYTES // (1024*1024)}MB limit.",
        )

    # Pass the CLEANED base_mime_type to Gemini instead of the messy one
    result = await run_in_threadpool(transcribe_audio, audio_bytes, base_mime_type)
    return result