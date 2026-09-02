#!/usr/bin/env python3
"""
intent_extractor.py
====================

Step 1 of the Real Estate Property Finder pipeline.

Implements the dual-model architecture: local Qwen (GGUF via llama_cpp
or HF transformers) is tried FIRST. If output is invalid/missing, falls
back to Gemini API.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional

from app.pipeline.schema_utils import (
    INTENT_SCHEMA,
    SYSTEM_PROMPT,
    is_schema_valid,
    normalize_intent,
    needs_clarification,
)

# --------------------------------------------------------------------------- #
# Dependencies -- imported lazily/defensively
# --------------------------------------------------------------------------- #
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
except ImportError:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    PeftModel = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("intent_extractor")

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"
QWEN_MAX_NEW_TOKENS = 256

# Module-level cache so Qwen is only loaded once per process
_QWEN_CACHE: dict[str, Any] = {}
QWEN_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _is_adapter_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "adapter_config.json"))


def _load_qwen(model_path: str) -> Optional[Any]:
    model_path = model_path.strip().strip('"').strip("'")
    if model_path in _QWEN_CACHE:
        return _QWEN_CACHE[model_path]

    if not os.path.exists(model_path):
        logger.warning("Qwen model path not found: %s -- Gemini-only mode.", model_path)
        _QWEN_CACHE[model_path] = None
        return None

    # --- Path A: GGUF Model via llama_cpp ---
    if os.path.isfile(model_path) or model_path.endswith(".gguf"):
        if Llama is None:
            logger.warning("llama-cpp-python not installed -- cannot load GGUF model. Falling back to Gemini.")
            _QWEN_CACHE[model_path] = None
            return None

        try:
            logger.info("Loading GGUF model via llama_cpp from %s ...", model_path)
            llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
            _QWEN_CACHE[model_path] = llm
            logger.info("Qwen GGUF model loaded successfully.")
            return llm
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load GGUF model from %s: %s -- Gemini-only mode.", model_path, exc)
            _QWEN_CACHE[model_path] = None
            return None

    # --- Path B: HuggingFace Directory / LoRA Adapter ---
    if torch is None or AutoModelForCausalLM is None:
        logger.warning("torch/transformers not installed -- cannot load HF model. Falling back to Gemini.")
        _QWEN_CACHE[model_path] = None
        return None

    is_adapter = _is_adapter_dir(model_path)

    try:
        if is_adapter:
            if PeftModel is None:
                logger.warning("peft not installed -- cannot load LoRA adapter. Falling back to Gemini.")
                _QWEN_CACHE[model_path] = None
                return None

            logger.info("Detected LoRA adapter directory: %s", model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            if torch.cuda.is_available() and BitsAndBytesConfig is not None:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    QWEN_BASE_MODEL, quantization_config=bnb_config, device_map="auto"
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(QWEN_BASE_MODEL)

            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            logger.info("Detected standalone/merged model directory: %s", model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        model.eval()
        _QWEN_CACHE[model_path] = (model, tokenizer)
        logger.info("Qwen HF model loaded successfully.")
        return (model, tokenizer)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load Qwen model from %s: %s -- Gemini-only mode.", model_path, exc)
        _QWEN_CACHE[model_path] = None
        return None


def _generate_with_qwen(loaded_obj: Any, query: str) -> str:
    """Runs generation with either llama_cpp or Transformers."""
    # GGUF / llama_cpp instance
    if Llama is not None and isinstance(loaded_obj, Llama):
        response = loaded_obj.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=QWEN_MAX_NEW_TOKENS,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"].strip()

    # Hugging Face (model, tokenizer) tuple
    model, tokenizer = loaded_obj
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Gemini (fallback) path
# --------------------------------------------------------------------------- #
def _generate_with_gemini(query: str, api_key: str) -> Optional[str]:
    if genai is None:
        logger.error("google-genai SDK is not installed. Run: pip install google-genai")
        return None
    if not api_key:
        logger.error("No Gemini API key provided.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=query,
            config=config,
        )
        return _extract_response_text(response)
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini API call failed for query %r: %s", query, exc)
        return None


def _extract_response_text(response: Any) -> Optional[str]:
    try:
        text = response.text
        return text.strip() if text else None
    except Exception:  # noqa: BLE001
        try:
            for candidate in getattr(response, "candidates", None) or []:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []):
                    part_text = getattr(part, "text", None)
                    if part_text:
                        return part_text.strip()
        except Exception:  # noqa: BLE001
            pass
    return None


# --------------------------------------------------------------------------- #
# Core routing logic
# --------------------------------------------------------------------------- #
def _try_parse_json(raw_text: Optional[str]) -> Optional[dict]:
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def route_intent(
    query: str,
    qwen_generate_fn=None,
    gemini_generate_fn=None,
) -> tuple[dict, str]:
    if not query or not query.strip():
        return dict(INTENT_SCHEMA), "empty"

    if qwen_generate_fn is not None:
        raw = qwen_generate_fn(query)
        parsed = _try_parse_json(raw)
        if parsed is not None:
            valid, reason = is_schema_valid(parsed)
            if valid:
                logger.info("Qwen succeeded (schema-valid).")
                return normalize_intent(parsed, raw_query=query), "qwen"
            logger.info("Qwen output failed schema validation (%s) -- falling back to Gemini.", reason)
        else:
            logger.info("Qwen output was not valid JSON -- falling back to Gemini.")
    else:
        logger.debug("No Qwen model loaded -- using Gemini directly.")

    if gemini_generate_fn is not None:
        raw = gemini_generate_fn(query)
        parsed = _try_parse_json(raw)
        if parsed is not None:
            logger.info("Gemini succeeded.")
            return normalize_intent(parsed, raw_query=query), "gemini"
        logger.error("Gemini output was not valid JSON either -- returning empty intent.")

    return dict(INTENT_SCHEMA), "empty"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_query_intent(
    query: str,
    gemini_api_key: str,
    qwen_path: Optional[str] = None,
) -> dict:
    qwen_generate_fn = None
    if qwen_path:
        loaded = _load_qwen(qwen_path)
        if loaded is not None:
            qwen_generate_fn = lambda q: _generate_with_qwen(loaded, q)  # noqa: E731

    gemini_generate_fn = lambda q: _generate_with_gemini(q, gemini_api_key)  # noqa: E731

    intent, source = route_intent(query, qwen_generate_fn, gemini_generate_fn)
    logger.info("Intent source for query %r: %s", query, source)
    return intent


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    qwen_path = os.environ.get("QWEN_MODEL_PATH")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    test_queries = [
        "I want a cosy 5 marla house in DHA Phase 6 under 2.5 crore near a school and park.",
        "Looking for a 1 kanal plot in Bahria Town Rawalpindi.",
    ]

    for i, query in enumerate(test_queries, start=1):
        print(f"\n--- Test {i} ---")
        intent = extract_query_intent(query, gemini_api_key=api_key, qwen_path=qwen_path)
        print("Extracted Intent:", json.dumps(intent, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()