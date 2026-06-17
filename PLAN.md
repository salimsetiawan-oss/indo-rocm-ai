# 📋 IndoROC — Implementation Plan

> Indonesian AI on AMD ROCm: LLM Fine-Tuning + Document Vision Pipeline

---

## Project Summary

**Name:** IndoROC
**Goal:** Fine-tune Indonesian LLMs + Build Document Vision Pipeline on AMD Instinct GPUs
**Duration:** 4-6 weeks
**Hardware:** AMD Instinct MI250 / MI300X

---

## Milestone 1: Environment Setup & Validation (Week 1)

### Objectives
- Set up ROCm development environment on AMD Instinct hardware
- Validate PyTorch + ROCm integration
- Confirm GPU access and basic operations

### Tasks
- [ ] Provision AMD Instinct GPU instance (MI250 or MI300)
- [ ] Install ROCm 6.x toolkit and drivers
- [ ] Set up Python 3.10+ with virtual environment
- [ ] Install PyTorch with ROCm backend (`torch-rocm`)
- [ ] Install Hugging Face Transformers + PEFT
- [ ] Run GPU validation test (`torch.cuda.is_available()`)
- [ ] Test basic tensor operations on GPU
- [ ] Document setup process for reproducibility

### Deliverables
- `scripts/setup_rocm.sh` — Automated setup script
- `docs/setup_guide.md` — Step-by-step environment guide
- GPU validation log with specs (VRAM, compute units, etc.)

---

## Milestone 2: Data Preparation (Week 1-2)

### Objectives
- Curate and prepare Indonesian language datasets
- Prepare document image samples for vision pipeline

### Tasks — Track 1 (LLM)
- [ ] Download Indonesian datasets:
  - `indonlp/NusaTranslation` (machine translation)
  - `indonlp/NusaParagraph` (text generation)
  - `indonlp/NusaX` (sentiment analysis)
  - Indonesian Wikipedia dump (instruction tuning)
- [ ] Clean and format datasets for fine-tuning
- [ ] Create instruction-tuning format (Alpaca-style)
- [ ] Split into train/eval/test sets
- [ ] Upload processed datasets to HF Hub

### Tasks — Track 2 (Vision)
- [ ] Collect sample Indonesian document images:
  - KTP (ID card) samples — synthetic/anonymized
  - Faktur pajak (tax invoice) samples
  - Receipt/struk samples
- [ ] Create annotation schema for structured extraction
- [ ] Build preprocessing pipeline (resize, enhance, normalize)

### Deliverables
- `src/indo-llm/prepare_data.py` — Data preparation script
- `data/datasets.md` — Dataset documentation
- HuggingFace dataset repo: `username/indo-instruction-v1`
- `src/vision-pipeline/preprocess.py` — Image preprocessing

---

## Milestone 3: LLM Fine-Tuning on ROCm (Week 2-3)

### Objectives
- Fine-tune Llama 3 8B (or Mistral 7B) for Indonesian language
- Document ROCm training performance
- Optimize for memory and throughput

### Tasks
- [ ] Load base model (Llama 3 8B or Mistral 7B)
- [ ] Configure LoRA parameters:
  - Rank: 16-64 (test multiple)
  - Alpha: 32-128
  - Target modules: q_proj, v_proj, k_proj, o_proj
- [ ] Configure QLoRA (4-bit) if VRAM limited
- [ ] Training configuration:
  - Batch size: 4-16 (depending on VRAM)
  - Learning rate: 2e-4 with cosine scheduler
  - Epochs: 3-5
  - Gradient accumulation: 4
- [ ] Run fine-tuning on AMD Instinct GPU
- [ ] Monitor training loss, GPU utilization, memory usage
- [ ] Test different LoRA ranks for speed/quality tradeoff
- [ ] Merge adapter weights with base model
- [ ] Push fine-tuned model to HF Hub

### Deliverables
- `src/indo-llm/train.py` — Training script (ROCm-compatible)
- `configs/lora_config.yaml` — LoRA hyperparameters
- `configs/training_args.yaml` — Training arguments
- HuggingFace model: `username/indo-llama3-lora`
- Training logs with GPU metrics

---

## Milestone 4: LLM Evaluation & Benchmarking (Week 3-4)

### Objectives
- Evaluate fine-tuned model quality
- Benchmark ROCm performance (tokens/sec, VRAM)
- Compare with published CUDA benchmarks

### Tasks
- [ ] Evaluation on Indonesian benchmarks:
  - IndoNLI (natural language inference)
  - NusaX sentiment analysis
  - Translation quality (BLEU/COMET)
- [ ] Inference benchmark with vLLM:
  - Tokens/sec (prompt processing)
  - Tokens/sec (generation)
  - Time to first token (TTFT)
  - VRAM usage per batch size
- [ ] Compare with published NVIDIA A100/H100 numbers
- [ ] Test different quantization methods:
  - FP16 baseline
  - INT8 (GPTQ)
  - INT4 (AWQ/GGUF)
- [ ] Document optimization tips for ROCm

### Deliverables
- `src/indo-llm/evaluate.py` — Evaluation script
- `benchmarks/throughput_tests.py` — Benchmark automation
- `benchmarks/rocm_vs_cuda.md` — Performance comparison report
- `notebooks/02_training_analysis.ipynb` — Training analysis notebook
- Benchmark charts and tables

---

## Milestone 5: Vision Pipeline (Week 4-5)

### Objectives
- Build document understanding pipeline for Indonesian docs
- Combine vision model + LLM for structured extraction
- Create working demo

### Tasks
- [ ] Image preprocessing pipeline:
  - Deskew, denoise, contrast enhancement
  - Region of interest detection
- [ ] OCR integration:
  - PaddleOCR for Indonesian text
  - Tesseract as fallback
- [ ] Vision-Language model integration:
  - Florence-2 for image understanding
  - CLIP for image-text matching
- [ ] LLM integration for structured extraction:
  - Prompt engineering for JSON output
  - Field extraction: name, NIK, address (for KTP)
  - Field extraction: amount, date, items (for invoices)
- [ ] Build API endpoint (FastAPI)
- [ ] Create demo notebook

### Deliverables
- `src/vision-pipeline/extract.py` — Core extraction logic
- `src/vision-pipeline/pipeline.py` — Full pipeline
- `src/vision-pipeline/api.py` — FastAPI endpoint
- `notebooks/03_vision_demo.ipynb` — Interactive demo
- `docs/api_docs.md` — API documentation

---

## Milestone 6: Documentation & Publication (Week 5-6)

### Objectives
- Publish all deliverables
- Write ROCm porting guide
- Share with community

### Tasks
- [ ] Write ROCm porting guide for NLP practitioners:
  - Common CUDA → ROCm migration issues
  - Performance tips and tricks
  - Known limitations and workarounds
- [ ] Create project report:
  - Project overview and motivation
  - Technical approach
  - Results and benchmarks
  - Lessons learned
- [ ] Clean up all code and documentation
- [ ] Upload everything to GitHub
- [ ] Submit models to HF Hub
- [ ] Share with ROCm community / developer forums

### Deliverables
- `docs/rocm_porting_notes.md` — Porting guide
- `docs/project_report.md` — Full project report
- GitHub repository (public)
- HuggingFace models and datasets (public)

---

## Budget Estimation

| Milestone | GPU Hours | Est. Compute |
|-----------|-----------|-------------|
| M1: Environment Setup | 2-3 hrs | Small |
| M2: Data Preparation | 3-4 hrs | CPU only |
| M3: LLM Fine-Tuning | 10-15 hrs | Heavy GPU |
| M4: Evaluation & Benchmarks | 5-8 hrs | GPU |
| M5: Vision Pipeline | 5-8 hrs | GPU |
| M6: Documentation | 2-3 hrs | CPU/small |
| **Total** | **~30-40 hrs** | |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| ROCm compatibility issues | High | Start with known-working model (Llama 3), have Mistral as fallback |
| VRAM insufficient for 7B | Medium | Use QLoRA (4-bit), reduce batch size, gradient checkpointing |
| Training instability | Medium | Lower learning rate, use gradient clipping, save checkpoints frequently |
| Vision model not available on ROCm | Low | Use ONNX runtime as fallback, or CPU for vision + GPU for LLM |

---

## Success Criteria

- [ ] Fine-tuned Indonesian LLM published on HuggingFace
- [ ] Benchmark report showing ROCm throughput numbers
- [ ] Working document extraction demo (KTP + invoice)
- [ ] ROCm porting guide published
- [ ] All code open-sourced on GitHub

---

*Last updated: June 17, 2026*
