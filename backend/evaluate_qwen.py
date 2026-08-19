#!/usr/bin/env python3
"""
evaluate_qwen.py
===================

Measures the fine-tuned Qwen model's real accuracy on the held-out
validation set (val_split.jsonl, produced by finetune_qwen.py -- never
seen during training).

Reports TWO separate numbers, because they answer different questions:

  1. SCHEMA-VALIDITY RATE
     % of outputs that are well-formed JSON matching INTENT_SCHEMA.
     This is the number that maps directly to "how often do we avoid
     falling back to Gemini" -- schema validation failure is exactly
     the fallback trigger built into intent_extractor.py.

  2. FIELD-LEVEL ACCURACY
     Of the schema-valid outputs, how many fields actually match the
     ground truth exactly. A response can be perfectly well-formed and
     still be WRONG (e.g. extracted the wrong price). Schema-validity
     alone would hide that -- this is why we report both, not just one.

Usage:
    python evaluate_qwen.py --adapter ./qwen-intent-lora --val_data val_split.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("evaluate_qwen")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """\
You are a strict information-extraction engine for a Pakistani real estate \
search platform (similar to Zameen.com). Your ONLY job is to read a user's \
natural language property search query and convert it into a single JSON \
object that exactly matches the required schema. You do not chat, explain, \
apologize, or add any text outside the JSON object.

Required JSON keys: property_type (House/Flat/Farmhouse/Plot/null), \
subtype (specific listing subtype or null), location (string or null), \
marla_min (float or null), marla_max (float or null), price_max \
(integer PKR or null), soft_signals (list of strings), poi (list of \
strings). Convert Kanal to Marla (1 Kanal = 20 Marla) and Crore/Lakh to \
raw PKR (Crore = x10,000,000, Lakh = x100,000). Leave subtype null unless \
explicitly specified. Return ONLY the JSON object, nothing else."""

VALID_CATEGORIES = {"House", "Flat", "Farmhouse", "Plot", None}
SUBTYPES_BY_CATEGORY = {
    "House": {"House", "Upper Portion", "Lower Portion", "Room", None},
    "Flat": {"Flat", "Penthouse", None},
    "Farmhouse": {"Farmhouse", None},
    "Plot": {"Residential Plot", "Commercial Plot", "Agricultural Land", "Industrial Land", "Plot File", None},
    None: {None},
}
REQUIRED_KEYS = {"property_type", "subtype", "location", "marla_min", "marla_max", "price_max", "soft_signals", "poi"}


def is_schema_valid(resp) -> tuple[bool, str]:
    """Mirrors the exact validation intent_extractor.py's _normalize_intent
    would apply -- this determines whether Gemini fallback would fire."""
    if not isinstance(resp, dict):
        return False, "not a JSON object"
    missing = REQUIRED_KEYS - set(resp.keys())
    if missing:
        return False, f"missing keys: {missing}"
    cat = resp.get("property_type")
    if cat not in VALID_CATEGORIES:
        return False, f"invalid property_type: {cat!r}"
    sub = resp.get("subtype")
    if sub not in SUBTYPES_BY_CATEGORY.get(cat, set()):
        return False, f"subtype {sub!r} invalid for category {cat!r}"
    if not isinstance(resp.get("soft_signals"), list) or not isinstance(resp.get("poi"), list):
        return False, "soft_signals/poi not lists"
    for f in ("marla_min", "marla_max", "price_max"):
        v = resp.get(f)
        if v is not None and not isinstance(v, (int, float)):
            return False, f"{f} not numeric"
    return True, "ok"


def load_model(adapter_path: str):
    logger.info("Loading base model + adapter from %s...", adapter_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, query: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def field_match_score(predicted: dict, expected: dict) -> float:
    """Fraction of REQUIRED_KEYS where predicted exactly matches expected."""
    matches = 0
    for k in REQUIRED_KEYS:
        p, e = predicted.get(k), expected.get(k)
        if isinstance(p, list) and isinstance(e, list):
            if sorted(p) == sorted(e):
                matches += 1
        elif isinstance(p, (int, float)) and isinstance(e, (int, float)):
            if abs(p - e) < 1e-6:
                matches += 1
        elif p == e:
            matches += 1
    return matches / len(REQUIRED_KEYS)


def field_diffs(predicted: dict, expected: dict) -> dict:
    """Returns {field: (expected_value, predicted_value)} for every field
    that doesn't match -- this is what actually makes failures readable,
    instead of dumping the full JSON and making the reader reconstruct
    which specific field went wrong."""
    diffs = {}
    for k in REQUIRED_KEYS:
        p, e = predicted.get(k), expected.get(k)
        matched = False
        if isinstance(p, list) and isinstance(e, list):
            matched = sorted(p) == sorted(e)
        elif isinstance(p, (int, float)) and isinstance(e, (int, float)):
            matched = abs(p - e) < 1e-6
        else:
            matched = p == e
        if not matched:
            diffs[k] = (e, p)
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--val_data", default="val_split.jsonl")
    parser.add_argument("--target_schema_rate", type=float, default=0.90, help="The lead's target -- e.g. 0.90 for 90%")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)

    examples = []
    with open(args.val_data, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info("Evaluating on %d held-out examples...", len(examples))

    schema_valid_count = 0
    field_scores = []
    exact_match_count = 0
    failures = []
    mismatched_field_counts = Counter()

    for i, ex in enumerate(examples):
        query, expected = ex["query"], ex["response"]
        raw_output = generate(model, tokenizer, query)

        try:
            predicted = json.loads(raw_output)
        except json.JSONDecodeError:
            failures.append({"query": query, "expected": expected, "predicted_raw": raw_output, "reason": "not valid JSON at all", "diffs": {}})
            continue

        valid, reason = is_schema_valid(predicted)
        if valid:
            schema_valid_count += 1
            score = field_match_score(predicted, expected)
            field_scores.append(score)
            if score == 1.0:
                exact_match_count += 1
            else:
                diffs = field_diffs(predicted, expected)
                for field in diffs:
                    mismatched_field_counts[field] += 1
                failures.append({
                    "query": query, "expected": expected, "predicted": predicted,
                    "reason": f"schema OK but {len(diffs)} field(s) differ (score={score:.2f})",
                    "diffs": diffs,
                })
        else:
            failures.append({"query": query, "expected": expected, "predicted_raw": raw_output, "reason": f"schema INVALID: {reason}", "diffs": {}})

        if (i + 1) % 20 == 0:
            logger.info("Progress: %d/%d evaluated", i + 1, len(examples))

    n = len(examples)
    schema_rate = schema_valid_count / n if n else 0
    avg_field_score = sum(field_scores) / len(field_scores) if field_scores else 0
    exact_rate = exact_match_count / n if n else 0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total examples evaluated:        {n}")
    print(f"Schema-valid rate:                {schema_rate*100:.1f}%   <-- this is the 'avoids Gemini fallback' number")
    print(f"  (target: {args.target_schema_rate*100:.0f}%  ->  {'PASS' if schema_rate >= args.target_schema_rate else 'BELOW TARGET'})")
    print(f"Exact field match rate:            {exact_rate*100:.1f}%   (all 8 fields correct)")
    print(f"Average field-level accuracy:      {avg_field_score*100:.1f}%   (avg. fraction of fields correct, schema-valid outputs only)")

    if mismatched_field_counts:
        print(f"\n{'='*70}\nFIELD MISMATCH FREQUENCY (across all failures -- shows WHERE to focus)\n{'='*70}")
        for field, count in mismatched_field_counts.most_common():
            pct = 100 * count / n
            print(f"  {field:<15} {count:>4} failures  ({pct:.1f}% of all examples)")

    if failures:
        print(f"\n{'='*70}\nSAMPLE FAILURES (up to 15) -- expected vs predicted, field by field\n{'='*70}")
        for f in failures[:15]:
            print(f"\nQuery: {f['query']}")
            print(f"Reason: {f['reason']}")
            if f["diffs"]:
                for field, (expected_val, predicted_val) in f["diffs"].items():
                    print(f"  [{field}]  expected={expected_val!r}  ->  got={predicted_val!r}")
            else:
                print(f"  Raw output: {f.get('predicted_raw', '')[:200]}")

    print("\n" + "=" * 70)
    if schema_rate >= args.target_schema_rate:
        print(f"RESULT: Target met -- {schema_rate*100:.1f}% >= {args.target_schema_rate*100:.0f}% schema-valid.")
    else:
        gap = args.target_schema_rate - schema_rate
        print(f"RESULT: Below target by {gap*100:.1f} points. See recommendations below.")
        print(
            "\nNext steps if below target:\n"
            "  1. Look at the failure samples above -- is there a pattern (one category,\n"
            "     one field, one phrasing style)? Generate more targeted training examples\n"
            "     for that specific gap rather than more random examples.\n"
            "  2. If failures are evenly spread, consider more training epochs or a\n"
            "     slightly higher LoRA rank (try --lora_r 32).\n"
            "  3. If the training set is under ~1000 examples, generate more --\n"
            "     structured extraction tasks like this usually need volume more than\n"
            "     they need extra epochs on a small set."
        )


if __name__ == "__main__":
    main()
