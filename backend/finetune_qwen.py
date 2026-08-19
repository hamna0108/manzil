#!/usr/bin/env python3
"""
finetune_qwen.py
==================

QLoRA fine-tuning of Qwen2.5-1.5B-Instruct for the real estate intent
extraction task (Step 1 of the pipeline). Trained to directly replicate
the same behavior as the Gemini-based extract_query_intent() function in
intent_extractor.py, using the identical system prompt and JSON schema,
so the two models are interchangeable at inference time.

Install (run once, in a GPU runtime):
    !pip install -q -U transformers peft bitsandbytes accelerate trl datasets

Usage:
    python finetune_qwen.py \
        --data qwen_training_data.jsonl \
        --output_dir ./qwen-intent-lora \
        --epochs 3

Expects each line of --data to be a JSON object:
    {"query": "...", "response": {<INTENT_SCHEMA fields>}}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import inspect

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from data_utils import load_training_examples

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("finetune_qwen")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Identical to the system prompt in intent_extractor.py -- training and
# inference MUST use the same instructions, or the fine-tuned model will
# behave inconsistently with what search_engine.py expects at runtime.
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


def build_chat_text(tokenizer, query: str, response: dict) -> str:
    """
    Format one training example using Qwen's chat template:
    system prompt + user query -> assistant's JSON response.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to qwen_training_data.jsonl")
    parser.add_argument("--output_dir", default="./qwen-intent-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction held out for validation during training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--merge_and_save", action="store_true", help="Also save a merged full model (larger, easier to deploy)")
    args = parser.parse_args()

    random.seed(args.seed)

    # --- Load and split data -------------------------------------------
    examples = load_training_examples(args.data)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_split))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]
    logger.info("Train: %d examples, Val (held out during training): %d examples", len(train_examples), len(val_examples))

    # Save the val split to disk so evaluate_qwen.py can use the EXACT
    # same held-out set later -- never evaluate on data the model trained on.
    # Written inside output_dir (which should point at Drive) rather than
    # the local working directory, since Colab's local disk is wiped on
    # session reset but output_dir survives across sessions.
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    val_split_path = os.path.join(args.output_dir, "val_split.jsonl")
    with open(val_split_path, "w", encoding="utf-8") as fh:
        for ex in val_examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Held-out validation set saved to %s", val_split_path)

    # --- Load model in 4-bit (QLoRA) ------------------------------------
    logger.info("Loading base model %s in 4-bit...", MODEL_NAME)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config -----------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen2 attention + MLP projection layer names.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Build datasets ----------------------------------------------------
    def to_text(ex):
        return {"text": build_chat_text(tokenizer, ex["query"], ex["response"])}

    train_ds = Dataset.from_list(train_examples).map(to_text, remove_columns=["query", "response"])
    val_ds = Dataset.from_list(val_examples).map(to_text, remove_columns=["query", "response"])

    logger.info("Example formatted training text:\n%s", train_ds[0]["text"])

    # --- Train ---------------------------------------------------------
    # trl's SFTConfig has changed its accepted field names across
    # versions (warmup_ratio, dataset_text_field, max_seq_length,
    # eval_strategy have all shifted at one point or another). Rather
    # than crash on whichever field this installed version renamed,
    # build the full desired config, filter it down to whatever this
    # version's SFTConfig actually accepts, and warn about anything
    # dropped -- functionality degrades gracefully instead of crashing.
    desired_config = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        warmup_steps=int(0.05 * (len(train_ds) / (args.batch_size * args.grad_accum)) * args.epochs),
        logging_steps=10,
        eval_strategy="steps",
        evaluation_strategy="steps",  # older trl/transformers name for the same thing
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        dataset_text_field="text",
        max_seq_length=512,  # queries + JSON responses are short; keeps training fast
        max_length=512,  # newer trl renamed max_seq_length -> max_length
        packing=False,
        seed=args.seed,
    )

    accepted_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    filtered_config = {k: v for k, v in desired_config.items() if k in accepted_params}
    dropped = set(desired_config.keys()) - set(filtered_config.keys())
    if dropped:
        logger.warning(
            "This trl version's SFTConfig doesn't accept: %s -- skipping them "
            "(likely renamed/removed in your installed trl version; training "
            "will proceed with defaults for these).",
            sorted(dropped),
        )

    sft_config = SFTConfig(**filtered_config)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_and_save:
        logger.info("Merging LoRA weights into the base model for standalone deployment...")
        merged_dir = args.output_dir.rstrip("/") + "-merged"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info("Merged model saved to %s", merged_dir)

    logger.info("Done. Run evaluate_qwen.py against val_split.jsonl next to measure real accuracy.")


if __name__ == "__main__":
    main()
