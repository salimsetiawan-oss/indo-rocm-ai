#!/usr/bin/env python3
"""
IndoROC: Indonesian LLM Fine-Tuning on AMD ROCm
Train script for LoRA/QLoRA fine-tuning on AMD Instinct GPUs.

Usage:
    python train.py --config ../configs/lora_config.yaml
    python train.py --model meta-llama/Meta-Llama-3-8B --dataset username/indo-instruction-v1
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_rocm():
    """Verify ROCm/AMD GPU availability."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"✅ GPU Available: {gpu_name}")
        logger.info(f"   VRAM: {gpu_memory:.1f} GB")
        logger.info(f"   ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
        return True
    else:
        logger.error("❌ No GPU detected! Check ROCm installation.")
        sys.exit(1)


def load_model_and_tokenizer(model_name: str, config: dict):
    """Load model with optional quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    # QLoRA quantization config
    if config.get("quantization", {}).get("enabled", False):
        logger.info("📦 Loading model with 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        logger.info("📦 Loading model in bfloat16 (full precision)")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if config.get("quantization", {}).get("enabled", False):
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False  # Required for gradient checkpointing
    return model, tokenizer


def setup_lora(model, config: dict):
    """Apply LoRA adapter to model."""
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=config["lora"]["target_modules"],
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"🔧 LoRA Applied:")
    logger.info(f"   Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
    logger.info(f"   Total: {total:,}")

    return model


def prepare_dataset(dataset_path: str, tokenizer, max_length: int = 2048):
    """Load and tokenize dataset."""
    logger.info(f"📊 Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)

    def tokenize_function(examples):
        # Format as instruction-following
        texts = []
        for instruction, input_text, output in zip(
            examples.get("instruction", [""] * len(examples["input"])),
            examples.get("input", examples.get("text", [""] * len(examples["instruction"]))),
            examples.get("output", examples.get("text", [""] * len(examples["instruction"])))
        ):
            if input_text:
                text = f"### Instruksi:\n{instruction}\n\n### Input:\n{input_text}\n\n### Jawaban:\n{output}"
            else:
                text = f"### Instruksi:\n{instruction}\n\n### Jawaban:\n{output}"
            texts.append(text + tokenizer.eos_token)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    logger.info(f"   Train samples: {len(tokenized_dataset['train'])}")
    if "validation" in tokenized_dataset:
        logger.info(f"   Eval samples: {len(tokenized_dataset['validation'])}")

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="IndoROC: Fine-tune LLM on AMD ROCm")
    parser.add_argument("--config", type=str, default="../configs/lora_config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    train_config = load_config("../configs/training_args.yaml")["training"]
    opt_config = load_config("../configs/training_args.yaml")["optimization"]

    # Check ROCm
    check_rocm()

    # Load model
    model_name = args.model or config["model"]["name"]
    logger.info(f"🚀 Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, config)

    # Apply LoRA
    model = setup_lora(model, config)

    # Prepare dataset
    dataset_path = args.dataset or "username/indo-instruction-v1"
    dataset = prepare_dataset(dataset_path, tokenizer, max_length=2048)

    # Training arguments
    output_dir = args.output or train_config["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs or train_config["num_train_epochs"],
        per_device_train_batch_size=args.batch_size or train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=args.lr or opt_config["learning_rate"],
        weight_decay=opt_config["weight_decay"],
        adam_beta1=opt_config["adam_beta1"],
        adam_beta2=opt_config["adam_beta2"],
        adam_epsilon=opt_config["adam_epsilon"],
        max_grad_norm=opt_config["max_grad_norm"],
        lr_scheduler_type=opt_config["lr_scheduler_type"],
        warmup_ratio=opt_config["warmup_ratio"],
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
        push_to_hub=True,
        hub_model_id=f"username/{Path(output_dir).name}",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=2048,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset["train"].select(range(500))),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Log GPU memory before training
    if torch.cuda.is_available():
        logger.info(f"💾 GPU Memory before training:")
        logger.info(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Train
    logger.info("🏋️ Starting training...")
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    training_duration = end_time - start_time

    logger.info(f"✅ Training completed in {training_duration}")

    # Save final model
    trainer.save_model()
    trainer.push_to_hub()

    # Log final metrics
    logger.info(f"📊 Final training loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")

    # Save training summary
    summary = {
        "model": model_name,
        "dataset": dataset_path,
        "epochs": args.epochs or train_config["num_train_epochs"],
        "training_duration": str(training_duration),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "final_loss": trainer.state.log_history[-1].get("loss", "N/A"),
    }

    import json
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("🎉 Done! Model saved and pushed to HuggingFace Hub.")


if __name__ == "__main__":
    main()
