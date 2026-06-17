# 🇮🇩 IndoROC: Indonesian AI on AMD ROCm

> Fine-tuning Indonesian Language Models + Document Vision Pipeline on AMD Instinct GPUs

## 📋 Overview

**IndoROC** is a research and development project that leverages AMD Developer Cloud (Instinct MI250/MI300) to:

1. **Fine-tune open-source LLMs** for Indonesian language tasks using ROCm + PyTorch
2. **Build a multi-modal document pipeline** for Indonesian documents (KTP, faktur, invoices, etc.)

This project aims to demonstrate that AMD Instinct GPUs are viable for non-English NLP and vision-language tasks, while contributing to the underrepresented Indonesian AI ecosystem.

## 🎯 Objectives

### Track 1: Indonesian LLM Fine-Tuning
- Fine-tune Llama 3 / Mistral / DeepSeek on Indonesian datasets using LoRA/QLoRA
- Benchmark training throughput and inference speed on ROCm vs. published CUDA results
- Release fine-tuned models + training scripts to Hugging Face Hub

### Track 2: Multi-Modal Document Understanding
- Build an image-to-text pipeline for Indonesian documents (KTP, faktur pajak, receipts)
- Combine vision models (CLIP, Florence-2) with LLM for structured extraction
- Deploy as a working demo with API endpoint

## 🏗️ Architecture

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
│  │      ▼      │     │  (JSON/Markdown)       │   │
│  │  HF Hub     │     │                        │   │
│  └─────────────┘     └──────────────────────┘   │
│                                                   │
├─────────────────────────────────────────────────┤
│  ROCm 6.x  │  PyTorch  │  vLLM  │  Transformers │
└─────────────────────────────────────────────────┘
```

## 📊 Benchmark Targets

| Metric | Target | Hardware |
|--------|--------|----------|
| LoRA fine-tune throughput | tokens/sec documented | MI250 / MI300 |
| Inference (vLLM) | tok/s for 7B model | MI250 / MI300 |
| VRAM usage (QLoRA 7B) | < 16GB | MI250 |
| KTP extraction accuracy | > 90% field-level | MI250 |
| Document OCR latency | < 2 sec/image | MI250 |

## 🛠️ Tech Stack

- **GPU:** AMD Instinct MI250 / MI300X (via AMD Developer Cloud)
- **Framework:** ROCm 6.x, PyTorch 2.x, Hugging Face Transformers
- **Training:** LoRA via PEFT, QLoRA via bitsandbytes (ROCm fork)
- **Inference:** vLLM (ROCm build), llama.cpp (ROCm backend)
- **Vision:** CLIP, Florence-2, PaddleOCR
- **Languages:** Python 3.10+

## 📂 Project Structure

```
indo-rocm-ai/
├── README.md                   # This file
├── PLAN.md                     # Detailed milestones & deliverables
├── configs/                    # Training & model configs
│   ├── lora_config.yaml
│   ├── training_args.yaml
│   └── vision_pipeline.yaml
├── src/
│   ├── indo-llm/              # Track 1: LLM fine-tuning
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── prepare_data.py
│   │   └── merge_adapter.py
│   └── vision-pipeline/       # Track 2: Document understanding
│       ├── extract.py
│       ├── preprocess.py
│       ├── pipeline.py
│       └── api.py
├── scripts/                    # Helper scripts
│   ├── setup_rocm.sh
│   ├── benchmark.sh
│   └── upload_to_hf.sh
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_benchmark_results.ipynb
├── benchmarks/                 # Benchmark results & scripts
│   ├── rocm_vs_cuda.md
│   ├── throughput_tests.py
│   └── results/
├── data/                       # Dataset configs & samples
│   ├── datasets.md
│   └── samples/
├── models/                     # Model artifacts (gitignored)
├── results/                    # Training outputs
└── docs/                       # Documentation
    ├── setup_guide.md
    ├── rocm_porting_notes.md
    └── api_docs.md
```

## 🚀 Quick Start

```bash
# 1. Setup ROCm environment
bash scripts/setup_rocm.sh

# 2. Prepare Indonesian dataset
python src/indo-llm/prepare_data.py --dataset indonlp/NusaTranslation

# 3. Fine-tune Llama 3 with LoRA
python src/indo-llm/train.py --config configs/lora_config.yaml

# 4. Run inference benchmark
python benchmarks/throughput_tests.py --model ./models/indo-llama3-lora

# 5. Test vision pipeline
python src/vision-pipeline/pipeline.py --image data/samples/ktp_sample.jpg
```

## 📈 Expected Deliverables

| # | Deliverable | Type |
|---|-------------|------|
| 1 | Fine-tuned Indonesian LLM (Llama 3 8B) | HuggingFace Model |
| 2 | ROCm training benchmark report | Markdown + Charts |
| 3 | Document extraction pipeline | Python Package |
| 4 | KTP/Invoice extraction demo | API + Notebook |
| 5 | ROCm porting guide for NLP | Tutorial |
| 6 | AMD Developer Cloud usage report | Blog Post |

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- AMD Developer Program for cloud GPU credits
- Indonesian NLP community (IndoNLP, NusaCrowd, NusaTranslation)
- Hugging Face for model hosting & tools
- ROCm open-source community
