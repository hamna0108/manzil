#!/usr/bin/env python3
"""
remerge_qwen_adapter.py
==========================

Produces a CORRECT merged, standalone Qwen model for production
deployment -- fixing the earlier merge, which likely degraded quality by
merging LoRA weights directly into a 4-bit quantized base model (a known
QLoRA pitfall: quantized weights aren't simple floats you can cleanly
add a LoRA delta to, so the merge can silently corrupt quality without
erroring).

The fix: load the base model in FULL PRECISION (bf16, no quantization),
load the adapter on top of that clean base, merge, then save. This is
the standard documented approach and avoids any quantization-related
merge artifacts entirely.

Usage:
    python remerge_qwen_adapter.py \
        --adapter_path /content/drive/MyDrive/property_finder/qwen-intent-lora-v2 \
        --output_path /content/drive/MyDrive/property_finder/qwen-intent-merged-v2-fixed
"""

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("remerge_qwen_adapter")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True, help="Path to the VERIFIED-GOOD LoRA adapter directory (the one evaluate_qwen.py graded)")
    parser.add_argument("--output_path", required=True, help="Where to save the new, correctly-merged standalone model")
    args = parser.parse_args()

    logger.info("Loading base model %s in FULL PRECISION (no quantization)...", BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    logger.info("Loading adapter from %s onto the clean full-precision base...", args.adapter_path)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    logger.info("Merging LoRA weights into the full-precision base (this is the correct order)...")
    merged_model = model.merge_and_unload()

    logger.info("Saving correctly-merged model to %s ...", args.output_path)
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    logger.info("Done. Verify this new model matches the adapter's behavior before deploying it --")
    logger.info("run diagnose_model_mismatch.py again, pointing --merged_path at %s", args.output_path)


if __name__ == "__main__":
    main()
