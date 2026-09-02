from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    transcript: str
    no_speech_detected: bool = False