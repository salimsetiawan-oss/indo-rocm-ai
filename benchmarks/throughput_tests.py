#!/usr/bin/env python3
"""
IndoROC: Benchmark Script for ROCm GPU Performance
Tests training throughput, inference speed, and VRAM usage.

Usage:
    python benchmark.py --model meta-llama/Meta-Llama-3-8B
    python benchmark.py --model ./models/indo-llama3-lora --inference-only
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"error": "No GPU available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_total_gb": props.total_mem / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "rocm_version": getattr(torch.version, 'hip', 'N/A'),
        "pytorch_version": torch.__version__,
    }


def benchmark_matmul(size: int = 4096, iterations: int = 100) -> Dict:
    """Benchmark matrix multiplication throughput."""
    logger.info(f"🔢 Benchmarking matrix multiplication ({size}x{size}, {iterations} iters)...")

    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * size**3 * iterations  # FLOPs for matrix multiply
    tflops = flops / elapsed / 1e12

    return {
        "matrix_size": size,
        "iterations": iterations,
        "total_time_sec": round(elapsed, 3),
        "avg_time_ms": round(elapsed / iterations * 1000, 3),
        "tflops": round(tflops, 2),
    }


def benchmark_training(model_name: str, batch_size: int = 4, seq_len: int = 512) -> Dict:
    """Benchmark training throughput (tokens/sec)."""
    logger.info(f"🏋️ Benchmarking training: {model_name} (batch={batch_size}, seq={seq_len})")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.train()

    # Create dummy input
    dummy_text = "Ini adalah contoh teks bahasa Indonesia untuk benchmark. " * 50
    inputs = tokenizer(
        [dummy_text] * batch_size,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    ).to("cuda")

    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # Warmup
    for _ in range(3):
        outputs = model(**inputs, labels=labels)
        outputs.loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # Benchmark
    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        outputs = model(**inputs, labels=labels)
        outputs.loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = batch_size * seq_len * iterations
    tokens_per_sec = total_tokens / elapsed

    return {
        "model": model_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "iterations": iterations,
        "total_time_sec": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "vram_used_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "loss": round(outputs.loss.item(), 4),
    }


def benchmark_inference(model_name: str, prompt: str = None, max_tokens: int = 256) -> Dict:
    """Benchmark inference speed (tokens/sec)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    if prompt is None:
        prompt = "Jelaskan apa itu AMD ROCm dan mengapa penting untuk AI:"

    logger.info(f"⚡ Benchmarking inference: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Prompt processing (prefill) benchmark
    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(**inputs)
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - start

    # Generation benchmark
    with torch.no_grad():
        start = time.perf_counter()
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
        )
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - start

    generated_tokens = generated.shape[1] - inputs["input_ids"].shape[1]
    prompt_tokens = inputs["input_ids"].shape[1]

    return {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_time_sec": round(prefill_time, 3),
        "prefill_tokens_per_sec": round(prompt_tokens / prefill_time, 1),
        "generation_time_sec": round(gen_time, 3),
        "generation_tokens_per_sec": round(generated_tokens / gen_time, 1),
        "ttft_sec": round(prefill_time, 3),  # Time to first token
        "vram_used_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="IndoROC GPU Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu": get_gpu_info(),
        "benchmarks": {},
    }

    # Matmul benchmark
    results["benchmarks"]["matmul"] = benchmark_matmul()

    # Training benchmark (unless inference-only)
    if not args.inference_only:
        results["benchmarks"]["training"] = benchmark_training(
            args.model, args.batch_size, args.seq_len
        )

    # Inference benchmark
    results["benchmarks"]["inference"] = benchmark_inference(
        args.model, max_tokens=args.max_tokens
    )

    # Print results
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved to: {args.output}")

    # Print summary
    print("\n📋 SUMMARY:")
    print(f"   GPU: {results['gpu']['name']}")
    if "matmul" in results["benchmarks"]:
        print(f"   MatMul TFLOPS: {results['benchmarks']['matmul']['tflops']}")
    if "training" in results["benchmarks"]:
        print(f"   Training: {results['benchmarks']['training']['tokens_per_sec']} tok/s")
        print(f"   VRAM Used: {results['benchmarks']['training']['vram_used_gb']} GB")
    if "inference" in results["benchmarks"]:
        print(f"   Inference: {results['benchmarks']['inference']['generation_tokens_per_sec']} tok/s")
        print(f"   TTFT: {results['benchmarks']['inference']['ttft_sec']}s")


if __name__ == "__main__":
    main()
