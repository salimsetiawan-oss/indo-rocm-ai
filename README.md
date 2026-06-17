# 🇮🇩 IndoROC: Indonesian AI on AMD ROCm

> Fine-tuning Indonesian Language Models + Document Vision Pipeline on AMD Instinct GPUs

## Overview

**IndoROC** is an open-source project exploring AMD Instinct GPUs (MI250/MI300) for underrepresented language AI — specifically Indonesian (Bahasa Indonesia).

Two tracks:
1. **Indonesian LLM Fine-Tuning** — LoRA/QLoRA on Llama 3 / Mistral for Indonesian NLP tasks
2. **Document Vision Pipeline** — Multi-modal extraction for KTP, faktur pajak, and receipts

The goal is simple: prove ROCm works for non-English NLP, publish benchmarks, and give the Indonesian AI community real tools.

## Why This Matters

- Indonesian is spoken by 270M+ people but massively underrepresented in LLM research
- AMD ROCm ecosystem lacks real-world NLP benchmarks outside English
- Document understanding for Indonesian bureaucracy (KTP, NPWP, faktur) is a genuine pain point
- Most GPU compute assumes CUDA — we need to validate alternatives

## Architecture

```
┌─────────────────────────────────────────────────┐
│              AMD Instinct GPU (ROCm)             │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────┐     ┌──────────────────────┐   │
│  │  Track 1    │     │  Track 2              │   │
│  │  LLM        │     │  Vision Pipeline      │   │
│  │             │     │                        │   │
│  │  Llama 3 ───┤     │  Image ──► CLIP ──┐   │   │
│  │  Mistral    │     │                    ├──►│   │
│  │  DeepSeek   │     │  KTP/Invoice ──►   │   │   │
│  │      │      │     │  Florence-2 ───────┘   │   │
│  │      ▼      │     │       │                │   │
│  │  LoRA/QLoRA │     │       ▼                │   │
│  │      │      │     │  Structured Output     │   │
│  │      ▼      │     │  (JSON)                │   │
│  │  HF Hub     │     │                        │   │
│  └─────────────┘     └──────────────────────┘   │
│                                                   │
├─────────────────────────────────────────────────┤
│  ROCm 6.x  │  PyTorch  │  vLLM  │  Transformers │
└─────────────────────────────────────────────────┘
```

## Benchmark Targets

| Metric | Target | Hardware |
|--------|--------|----------|
| LoRA fine-tune throughput | tokens/sec documented | MI250 / MI300 |
| Inference (vLLM) | tok/s for 7B model | MI250 / MI300 |
| VRAM usage (QLoRA 7B) | < 16GB | MI250 |
| KTP extraction accuracy | > 90% field-level | MI250 |
| Document OCR latency | < 2 sec/image | MI250 |

## Tech Stack

- **GPU:** AMD Instinct MI250 / MI300X
- **Framework:** ROCm 6.x, PyTorch 2.x, Hugging Face Transformers
- **Training:** LoRA via PEFT, QLoRA via bitsandbytes (ROCm)
- **Inference:** vLLM (ROCm build), llama.cpp (ROCm backend)
- **Vision:** CLIP, Florence-2, PaddleOCR
- **Languages:** Python 3.10+

## Project Structure

```
indo-rocm-ai/
├── README.md
├── PLAN.md
├── configs/
│   ├── lora_config.yaml
│   └── training_args.yaml
├── src/
│   ├── indo-llm/
│   │   ├── train.py              # Fine-tuning script
│   │   └── prepare_data.py       # Dataset preparation
│   └── vision-pipeline/
│       └── pipeline.py           # Document extraction
├── scripts/
│   └── setup_rocm.sh             # Environment setup
├── benchmarks/
│   └── throughput_tests.py       # GPU benchmarks
└── docs/
    └── rocm_porting_notes.md     # CUDA → ROCm guide
```

## Quick Start

```bash
# Setup ROCm environment
bash scripts/setup_rocm.sh

# Prepare Indonesian dataset
python src/indo-llm/prepare_data.py --dataset indonlp/NusaTranslation

# Fine-tune Llama 3 with LoRA
python src/indo-llm/train.py --config configs/lora_config.yaml

# Run inference benchmark
python benchmarks/throughput_tests.py --model ./models/indo-llama3-lora

# Test vision pipeline
python src/vision-pipeline/pipeline.py --image data/samples/ktp_sample.jpg
```

## Deliverables

| # | Deliverable | Type |
|---|-------------|------|
| 1 | Fine-tuned Indonesian LLM (Llama 3 8B) | HuggingFace Model |
| 2 | ROCm training benchmark report | Markdown + Charts |
| 3 | Document extraction pipeline | Python Package |
| 4 | KTP/Invoice extraction demo | API + Notebook |
| 5 | ROCm porting guide for NLP | Tutorial |

## Datasets

- [NusaTranslation](https://huggingface.co/datasets/indonlp/NusaTranslation) — Parallel translation
- [NusaX](https://huggingface.co/datasets/indonlp/NusaX) — Sentiment analysis
- [NusaParagraph](https://huggingface.co/datasets/indonlp/NusaParagraph) — Text generation

## License

MIT — See [LICENSE](LICENSE)

## Acknowledgments

- Indonesian NLP community (IndoNLP, NusaCrowd, NusaTranslation)
- Hugging Face for model hosting & tools
- ROCm open-source community
