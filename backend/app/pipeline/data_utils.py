#!/usr/bin/env python3
"""
data_utils.py
===============

Pure-Python data loading/normalization for the Qwen training pipeline.
Deliberately has ZERO dependencies beyond the standard library (no
torch, transformers, peft, trl) so that lightweight tasks -- like
re-splitting a JSONL file -- don't require the entire GPU/ML stack to
be installed just to shuffle a list.

Used by:
    - finetune_qwen.py (needs the heavy ML stack anyway, for training)
    - regenerate_val_split.py (does NOT need the ML stack -- this is
      exactly why this module exists as a separate file)
    - validate_training_data.py (has its own copy for zero-dependency
      standalone use, kept deliberately in sync with this one)
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("data_utils")


def normalize_record(raw: dict, line_num: int) -> dict | None:
    """
    Accepts either shape and returns a normalized {"query": ..., "response": ...}
    dict, or None if the record can't be normalized.

    Shape A (simple):
        {"query": "...", "response": {...}}

    Shape B (ChatML, what qwen_training_data.jsonl actually uses):
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "<query>"},
            {"role": "assistant", "content": "<JSON string>"}
        ]}

    Note: any embedded 'system' message in Shape B is intentionally
    IGNORED by callers that build their own chat text (e.g.
    finetune_qwen.py always applies its own canonical SYSTEM_PROMPT),
    so training stays consistent with intent_extractor.py regardless of
    what prompt the data was originally generated with.
    """
    if "query" in raw and "response" in raw:
        return {"query": raw["query"], "response": raw["response"]}

    if "messages" in raw and isinstance(raw["messages"], list):
        user_msg = next((m for m in raw["messages"] if m.get("role") == "user"), None)
        assistant_msg = next((m for m in raw["messages"] if m.get("role") == "assistant"), None)
        if user_msg is None or assistant_msg is None:
            logger.warning("line %d: messages missing user/assistant -- skipping", line_num)
            return None

        query = user_msg.get("content", "")
        cleaned = assistant_msg.get("content", "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            response = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("line %d: assistant content not valid JSON -- skipping", line_num)
            return None
        return {"query": query, "response": response}

    logger.warning("line %d: unrecognized record shape -- skipping", line_num)
    return None


def load_training_examples(path: str) -> list[dict]:
    examples = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            norm = normalize_record(raw, i)
            if norm is None:
                skipped += 1
                continue
            examples.append(norm)
    logger.info("Loaded %d training examples from %s (%d skipped)", len(examples), path, skipped)
    if not examples:
        raise ValueError(f"No usable examples found in {path} -- check the warnings above.")
    return examples
