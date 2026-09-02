"""
Voice-to-text for Manzil's search bar, via Gemini's audio understanding
(not the browser's free Web Speech API) -- chosen specifically because
users describe listings in code-switched English/Urdu/Roman Urdu
("5 marla ghar chahiye DHA mein"), which browser speech APIs handle
poorly and inconsistently across browsers. Gemini already sits in our
stack for intent extraction and summaries, so this reuses infrastructure
rather than adding a new vendor.

Design principle carried over from the rest of this pipeline: this
TRANSCRIBES only, it does not interpret or answer the query -- the
transcribed text is meant to land in the EXISTING editable search box
exactly like typed text would, so a mis-transcribed word is always
visible and correctable, never silently trusted.
"""

from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger("voice_service")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

settings = get_settings()

TRANSCRIBE_MODEL_NAME = "gemini-3.5-flash-lite"

# Inline audio requests are capped at 20MB TOTAL (text + audio) by the
# Gemini API itself. Voice search queries should be a few seconds long,
# so 8MB is already generous -- well over a minute of compressed speech
# -- and leaves real margin under that ceiling rather than pushing
# right up against it.
MAX_AUDIO_SIZE_BYTES = 8 * 1024 * 1024

# Mime types Gemini's audio understanding is documented to support.
# NOTE FOR THE FRONTEND (later): browsers' MediaRecorder API commonly
# defaults to audio/webm, which is NOT in this confirmed-supported
# list. When building the recording UI, request the recorder target
# audio/ogg or audio/wav explicitly (e.g.
# `new MediaRecorder(stream, { mimeType: 'audio/ogg;codecs=opus' })`),
# or add server-side transcoding before this point. Rejecting early
# here with a clear error is safer than silently sending an
# unsupported format to Gemini and getting a confusing failure.
SUPPORTED_MIME_TYPES = {
    "audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav",
    "audio/aac", "audio/aiff", "audio/ogg", "audio/flac",
    "audio/webm", "audio/mp4"  # <-- Added these two
}

NO_SPEECH_SENTINEL = "NO_SPEECH_DETECTED"

TRANSCRIBE_SYSTEM_PROMPT = f"""\
You are a speech transcription engine for a Pakistani real estate search \
platform. You will be given a short audio clip of a user describing the \
property they're looking for.

Transcribe EXACTLY what was said. Rules:
- Do NOT translate. If the speaker mixes English and Urdu/Roman Urdu \
("5 marla ghar chahiye DHA mein"), transcribe it exactly as a natural \
Roman Urdu / English mix, the same way the speaker said it. Do not \
convert it into pure English or pure Urdu script.
- Do NOT answer, interpret, summarize, or respond to the query in any \
way. Your only job is transcription.
- Do NOT add commentary, labels, timestamps, or speaker names. Return \
ONLY the transcribed text.
- Expect real estate vocabulary: marla, kanal, crore, lakh, lac, DHA, \
Bahria Town, Gulberg, Askari, plot, farmhouse, and Pakistani city names. \
Prioritize these interpretations for ambiguous-sounding words when they \
fit the context of a property search.
- If the audio is silent, contains no discernible speech, or is pure \
noise, respond with EXACTLY this string and nothing else: {NO_SPEECH_SENTINEL}
"""


def transcribe_audio(audio_bytes: bytes, mime_type: str) -> dict:
    """
    Synchronous, blocking Gemini call -- callers (the router) MUST wrap
    this in run_in_threadpool, consistent with every other Gemini/
    network call in this pipeline (see services/pipeline_bridge.py).

    Never raises for a "normal" failure (missing SDK, no key, API
    error) -- returns a graceful empty-transcript result instead, same
    principle as every other AI call in this project: a failed voice
    transcription should never crash the search, it should just fall
    back to an empty search box the user can type into.
    """
    if genai is None:
        logger.error("google-genai SDK is not installed.")
        return {"transcript": "", "no_speech_detected": False}
    if not settings.gemini_api_key:
        logger.error("No Gemini API key configured.")
        return {"transcript": "", "no_speech_detected": False}

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        config = genai_types.GenerateContentConfig(system_instruction=TRANSCRIBE_SYSTEM_PROMPT)
        response = client.models.generate_content(
            model=TRANSCRIBE_MODEL_NAME,
            contents=[
                "Transcribe this audio clip.",
                genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
            config=config,
        )
        text = (getattr(response, "text", None) or "").strip()

        if not text or text == NO_SPEECH_SENTINEL:
            return {"transcript": "", "no_speech_detected": True}
        return {"transcript": text, "no_speech_detected": False}

    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini audio transcription failed: %s", exc)
        return {"transcript": "", "no_speech_detected": False}