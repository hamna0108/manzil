#!/usr/bin/env python3
"""
merge_training_batches.py
============================

Combines multiple training data files (the original qwen_training_data.jsonl
plus the new targeted batch) into one deduplicated file, ready for
finetune_qwen.py. Handles mixed input shapes automatically:
  - A JSON array file (what Gemini outputs directly, e.g. batch2_raw.json)
  - A JSONL file in {"query","response"} shape
  - A JSONL file in ChatML {"messages": [...]} shape (the original file)

Deduplicates by exact query text (case-insensitive, whitespace-normalized)
across ALL input files combined, keeping the FIRST occurrence.

Usage:
    python merge_training_batches.py \
        --inputs qwen_training_data.jsonl batch2_raw.json \
        --output qwen_training_data_v2.jsonl
"""

import argparse
import json

from data_utils import normalize_record


def load_any_format(path: str) -> list[dict]:
    """Load either a JSON array file or a JSONL file, returning raw dicts
    (not yet normalized)."""
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read().strip()

    # Try JSON array first (what a raw Gemini batch output looks like)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (one object per line)
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to all input files, in priority order (earlier files win on duplicates)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seen_queries = set()
    merged = []
    total_loaded = 0
    skipped_unparseable = 0
    duplicates_removed = 0

    for path in args.inputs:
        raw_records = load_any_format(path)
        print(f"{path}: {len(raw_records)} raw record(s)")

        for i, raw in enumerate(raw_records, start=1):
            total_loaded += 1
            norm = normalize_record(raw, i)
            if norm is None:
                skipped_unparseable += 1
                continue

            key = " ".join(norm["query"].strip().lower().split())
            if key in seen_queries:
                duplicates_removed += 1
                continue

            seen_queries.add(key)
            merged.append(norm)

    with open(args.output, "w", encoding="utf-8") as fh:
        for ex in merged:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Total records loaded:      {total_loaded}")
    print(f"Skipped (unparseable):     {skipped_unparseable}")
    print(f"Duplicates removed:        {duplicates_removed}")
    print(f"Final unique examples:     {len(merged)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
