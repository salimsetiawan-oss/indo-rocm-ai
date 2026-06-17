#!/usr/bin/env python3
"""
IndoROC: Data Preparation Script
Prepare Indonesian language datasets for fine-tuning.

Supported datasets:
- indonlp/NusaTranslation
- indonlp/NusaX
- indonlp/NusaParagraph
- Custom instruction datasets

Usage:
    python prepare_data.py --dataset indonlp/NusaTranslation
    python prepare_data.py --dataset indonlp/NusaX --task sentiment
    python prepare_data.py --custom --input data/raw/ --output data/processed/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_nusa_translation(output_dir: str = "data/processed"):
    """Prepare NusaTranslation dataset for instruction tuning."""
    logger.info("📊 Loading NusaTranslation dataset...")
    dataset = load_dataset("indonlp/NusaTranslation")

    # Convert to instruction format
    def format_translation(examples):
        instructions = []
        inputs_list = []
        outputs = []

        for src, tgt, src_lang, tgt_lang in zip(
            examples["text"], examples["label"],
            examples.get("lang_src", ["id"] * len(examples["text"])),
            examples.get("lang_tgt", ["en"] * len(examples["text"]))
        ):
            instructions.append(f"Terjemahkan teks berikut dari {src_lang} ke {tgt_lang}:")
            inputs_list.append(src)
            outputs.append(tgt)

        return {
            "instruction": instructions,
            "input": inputs_list,
            "output": outputs,
        }

    processed = dataset.map(format_translation, batched=True, remove_columns=dataset["train"].column_names)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    processed.save_to_disk(os.path.join(output_dir, "nusa_translation"))

    logger.info(f"✅ Saved to {output_dir}/nusa_translation")
    logger.info(f"   Train: {len(processed['train'])}")
    if "validation" in processed:
        logger.info(f"   Valid: {len(processed['validation'])}")

    return processed


def prepare_nusax(output_dir: str = "data/processed"):
    """Prepare NusaX dataset for sentiment analysis instruction tuning."""
    logger.info("📊 Loading NusaX dataset...")
    dataset = load_dataset("indonlp/NusaX", "ind")

    label_map = {
        0: "negatif",
        1: "netral",
        2: "positif",
    }

    def format_sentiment(examples):
        instructions = []
        inputs_list = []
        outputs = []

        for text, label in zip(examples["text"], examples["label"]):
            instructions.append("Klasifikasikan sentimen teks berikut (positif/negatif/netral):")
            inputs_list.append(text)
            outputs.append(label_map.get(label, str(label)))

        return {
            "instruction": instructions,
            "input": inputs_list,
            "output": outputs,
        }

    processed = dataset.map(format_sentiment, batched=True, remove_columns=dataset["train"].column_names)

    os.makedirs(output_dir, exist_ok=True)
    processed.save_to_disk(os.path.join(output_dir, "nusax_sentiment"))

    logger.info(f"✅ Saved to {output_dir}/nusax_sentiment")
    return processed


def prepare_instruction_dataset(
    datasets_list: List[str],
    output_dir: str = "data/processed",
    max_samples: int = None,
):
    """Combine multiple datasets into a unified instruction dataset."""
    logger.info(f"📊 Preparing instruction dataset from {len(datasets_list)} sources...")

    all_data = []

    for ds_name in datasets_list:
        try:
            ds = load_dataset(ds_name)
            if "train" in ds:
                all_data.extend(ds["train"].to_list())
                logger.info(f"   ✅ {ds_name}: {len(ds['train'])} samples")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load {ds_name}: {e}")

    if max_samples and len(all_data) > max_samples:
        import random
        random.shuffle(all_data)
        all_data = all_data[:max_samples]

    # Convert to Dataset
    combined = Dataset.from_list(all_data)

    # Split
    split = combined.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(os.path.join(output_dir, "combined_instruction"))

    logger.info(f"✅ Combined dataset saved:")
    logger.info(f"   Train: {len(dataset_dict['train'])}")
    logger.info(f"   Valid: {len(dataset_dict['validation'])}")

    # Push to Hub if token available
    hub_token = os.environ.get("HF_TOKEN")
    if hub_token:
        hub_id = f"username/indo-instruction-v1"
        dataset_dict.push_to_hub(hub_id, token=hub_token)
        logger.info(f"📤 Pushed to HuggingFace Hub: {hub_id}")

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="IndoROC Data Preparation")
    parser.add_argument("--dataset", type=str, default="indonlp/NusaTranslation",
                        help="Dataset name on HuggingFace")
    parser.add_argument("--task", type=str, default="translation",
                        choices=["translation", "sentiment", "combined"])
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    if args.task == "translation":
        prepare_nusa_translation(args.output)
    elif args.task == "sentiment":
        prepare_nusax(args.output)
    elif args.task == "combined":
        prepare_instruction_dataset(
            ["indonlp/NusaTranslation", "indonlp/NusaX", "indonlp/NusaParagraph"],
            args.output,
            args.max_samples,
        )


if __name__ == "__main__":
    main()
