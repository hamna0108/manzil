#!/usr/bin/env python3
"""
intent_extractor.py
====================

Step 1 of the Real Estate Property Finder pipeline.

Implements the dual-model architecture: the fine-tuned Qwen2.5-1.5B model
is tried FIRST (fast, cheap, local). Its output is validated against the
schema; only if Qwen's output is missing/malformed does the pipeline fall
back to the Gemini API. This is the actual production routing logic that
evaluate_qwen.py's "schema-valid rate" metric was measuring in advance.

    query -> Qwen (if loaded) -> schema check -> valid?  -> use it, done
                                                -> invalid -> Gemini -> use it

If no Qwen model is loaded (qwen_path=None), the pipeline transparently
falls back to Gemini-only behavior -- this keeps the script fully usable
and testable before a fine-tuned model is available.

Usage:
    from intent_extractor import extract_query_intent, needs_clarification

    intent = extract_query_intent(
        query="5 marla house in DHA under 2 crore",
        gemini_api_key="...",
        qwen_path="/content/drive/MyDrive/property_finder/qwen-intent-lora-v2",  # ADAPTER dir, not "-merged"
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional

from schema_utils import (
    INTENT_SCHEMA,
    SYSTEM_PROMPT,
    is_schema_valid,
    normalize_intent,
    needs_clarification,
)

# --------------------------------------------------------------------------- #
# Optional dependencies -- both Qwen (torch/transformers) and Gemini
# (google-genai) are imported lazily/defensively, so this module works
# (in Gemini-only mode) even in an environment without a GPU stack, and
# vice versa.
# --------------------------------------------------------------------------- #
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

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

# Module-level cache so the (fairly expensive to load) Qwen model is only
# loaded once per process, even across many extract_query_intent() calls.
_QWEN_CACHE: dict[str, tuple[Any, Any]] = {}


# --------------------------------------------------------------------------- #
# Qwen (primary) path
# --------------------------------------------------------------------------- #
QWEN_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _is_adapter_dir(path: str) -> bool:
    """
    A LoRA adapter directory contains adapter_config.json (and
    adapter_model.safetensors); a full standalone/merged model directory
    contains config.json (and model.safetensors) instead. Checking this
    lets _load_qwen() handle either kind of directory correctly without
    the caller needing to remember which one they're pointing at --
    important for a deployed product where "load whatever model path is
    configured" needs to just work.
    """
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _load_qwen(model_path: str) -> Optional[tuple[Any, Any]]:
    """
    Load (and cache) the fine-tuned Qwen model + tokenizer from
    `model_path`, which may point at EITHER:
      - a LoRA adapter directory (base model + adapter loaded via
        PeftModel, 4-bit quantized when a GPU is available), or
      - a full standalone/merged model directory (loaded directly,
        no peft/bitsandbytes needed at all -- simpler and faster for
        production deployment).

    Which mode is used is auto-detected from the directory's contents.

    Background: an earlier merged model produced by
    finetune_qwen.py --merge_and_save was found (via
    diagnose_model_mismatch.py) to behave meaningfully worse than the
    adapter it was merged from. The likely cause: merging LoRA weights
    directly into a still-4-bit-quantized base model is a known QLoRA
    pitfall that can silently degrade quality. remerge_qwen_adapter.py
    fixes this by merging into a full-precision base instead -- once
    you've run that and verified the result, point model_path at the
    NEW merged directory for the simplest, fastest production setup.
    """
    if model_path in _QWEN_CACHE:
        return _QWEN_CACHE[model_path]

    if torch is None or AutoModelForCausalLM is None:
        logger.warning(
            "torch/transformers not installed -- cannot load Qwen. "
            "Falling back to Gemini-only mode."
        )
        _QWEN_CACHE[model_path] = None
        return None

    if not os.path.isdir(model_path):
        logger.warning("Qwen model path not found: %s -- Gemini-only mode.", model_path)
        _QWEN_CACHE[model_path] = None
        return None

    is_adapter = _is_adapter_dir(model_path)

    try:
        if is_adapter:
            if PeftModel is None:
                logger.warning("peft not installed -- cannot load LoRA adapter. Falling back to Gemini-only mode.")
                _QWEN_CACHE[model_path] = None
                return None

            logger.info("Detected LoRA adapter directory. Loading base model %s + adapter from %s ...", QWEN_BASE_MODEL, model_path)
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
            logger.info("Detected standalone/merged model directory. Loading directly from %s ...", model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        model.eval()
        _QWEN_CACHE[model_path] = (model, tokenizer)
        logger.info("Qwen model loaded successfully (%s).", "adapter" if is_adapter else "merged")
        return (model, tokenizer)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load Qwen model from %s: %s -- Gemini-only mode.", model_path, exc)
        _QWEN_CACHE[model_path] = None
        return None


def _generate_with_qwen(model, tokenizer, query: str) -> str:
    """Runs one deterministic generation and returns the raw decoded text."""
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
    """Calls Gemini and returns the raw text response, or None on failure."""
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
# Core routing logic (pure, dependency-injected -- see extract_query_intent
# for the real wrapper). Kept separate from I/O so it can be unit-tested
# without a GPU or an API key: see test_intent_extractor.py.
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
    """
    Pure routing logic: tries qwen_generate_fn first (if provided), falls
    back to gemini_generate_fn if Qwen's output is missing/unparseable/
    schema-invalid. Both *_generate_fn params are callables taking (query)
    and returning a raw text string (or None on failure) -- this
    indirection is what makes the routing logic testable without a real
    model or API key.

    Returns (intent_dict, source) where source is one of:
        "qwen", "gemini", "empty" (both paths failed / neither available)
    """
    if not query or not query.strip():
        return dict(INTENT_SCHEMA), "empty"

    if qwen_generate_fn is not None:
        raw = qwen_generate_fn(query)
        parsed = _try_parse_json(raw)
        if parsed is not None:
            valid, reason = is_schema_valid(parsed)
            if valid:
                logger.info("Qwen succeeded (schema-valid).")
                return normalize_intent(parsed), "qwen"
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
            return normalize_intent(parsed), "gemini"
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
    """
    Extract structured search intent from a natural language query.

    Tries the fine-tuned Qwen model first (if qwen_path is provided and
    loads successfully), falling back to Gemini if Qwen's output is
    missing, malformed, or fails schema validation. If qwen_path is None,
    uses Gemini directly (no behavior change from a Gemini-only setup).

    Args:
        query: The raw natural language search query.
        gemini_api_key: Gemini API key (used for the fallback path, or as
            the sole path if qwen_path is None).
        qwen_path: Path to the fine-tuned Qwen LoRA ADAPTER directory
            (e.g. the output_dir passed to finetune_qwen.py -- NOT a
            "-merged" directory; see _load_qwen()'s docstring for why).
            None to skip Qwen entirely and always use Gemini.

    Returns:
        A dict conforming to INTENT_SCHEMA. Never raises -- returns a
        safe all-null/empty-list default if both paths fail.
    """
    qwen_generate_fn = None
    if qwen_path:
        loaded = _load_qwen(qwen_path)
        if loaded is not None:
            model, tokenizer = loaded
            qwen_generate_fn = lambda q: _generate_with_qwen(model, tokenizer, q)  # noqa: E731

    gemini_generate_fn = lambda q: _generate_with_gemini(q, gemini_api_key)  # noqa: E731

    intent, source = route_intent(query, qwen_generate_fn, gemini_generate_fn)
    logger.info("Intent source for query %r: %s", query, source)
    return intent


# --------------------------------------------------------------------------- #
# Demo / manual test harness
# --------------------------------------------------------------------------- #
def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    qwen_path = os.environ.get("QWEN_MODEL_PATH")  # optional

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    test_queries = [
        "I want a cosy 5 marla house in DHA Phase 6 under 2.5 crore near a school and park.",
        "Looking for a 1 kanal plot in Bahria Town Rawalpindi.",
        "Modern apartment in Gulberg Lahore max 85 lakh with a balcony.",
        "4 kanal farmhouse in Chak Shahzad Islamabad, budget 5 crore.",
        "I want a commercial plot in DHA Lahore under 1 crore.",
        "DHA Lahore me 5 marla sasta ghar under 2 crore",
        "Gujranwala school ke pass ek 5 marla ghar sasta wala",
        "1.5 crore se 200 lakh tak ghar bahria me"
    ]

    logger.info(
        "=== Running intent_extractor.py demo (Qwen path: %s) ===",
        qwen_path or "not set -- Gemini only",
    )

    for i, query in enumerate(test_queries, start=1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")
        intent = extract_query_intent(query, gemini_api_key=api_key, qwen_path=qwen_path)
        print("Extracted Intent:")
        print(json.dumps(intent, indent=2, ensure_ascii=False))

        clarification = needs_clarification(intent)
        if clarification:
            print(f"[Clarification needed] {clarification['question']} Options: {clarification['options']}")
        else:
            print("[No clarification needed -- ready to search]")

    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    main()