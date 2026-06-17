# ROCm Porting Guide for NLP Practitioners

> Migrating CUDA-based NLP workloads to AMD ROCm

## Quick Reference

| CUDA | ROCm Equivalent |
|------|-----------------|
| `torch.cuda.is_available()` | Same (works with ROCm) |
| `torch.cuda.get_device_name()` | Same |
| CUDA 11.8 / 12.x | ROCm 5.7 / 6.0 |
| `bitsandbytes` | `bitsandbytes` (ROCm fork) |
| `flash-attn` | `flash-attn` (ROCm build, limited) |
| Triton | ROCm Triton support |
| NCCL | RCCL |

## Environment Setup

```bash
# Check ROCm version
rocm-smi --version

# Check GPU info
rocm-smi --showproductname

# Verify PyTorch sees the GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Set environment variables
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

## Common Issues & Solutions

### 1. `bitsandbytes` Installation

```bash
# Standard bitsandbytes doesn't work on ROCm
# Use the ROCm-compatible fork:
pip install bitsandbytes>=0.43.0
# Or build from source:
git clone https://github.com/ROCm/bitsandbytes.git
cd bitsandbytes
make hip ROCM_HOME=/opt/rocm
pip install .
```

### 2. Flash Attention

```bash
# Flash Attention has limited ROCm support
# Check compatibility:
pip install flash-attn --no-build-isolation
# If fails, fall back to standard attention:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",  # Use "eager" instead of "flash_attention_2"
)
```

### 3. `HSA_OVERRIDE_GFX_VERSION`

```bash
# Set correct GFX version for your GPU
# MI250:  export HSA_OVERRIDE_GFX_VERSION=9.0.0
# MI300X: export HSA_OVERRIDE_GFX_VERSION=9.4.0
# Check yours: rocm-smi --showproductname
```

### 4. Memory Management

```python
# ROCm memory management differs slightly
import torch

# Clear cache (same API)
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Use 8-bit optimizer
pip install bitsandbytes
# In training args: optim="paged_adamw_8bit"
```

### 5. DDP (Distributed Data Parallel)

```bash
# ROCm uses RCCL instead of NCCL
# PyTorch handles this automatically, but if issues:
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
```

### 6. Quantization

```python
# QLoRA with bitsandbytes on ROCm
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bf16, not fp16
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

## Performance Tips

1. **Use `bfloat16`** - MI250/MI300 have native bf16 support, often faster than fp16
2. **Use `torch.compile()`** - ROCm 6.0+ has good support
3. **Batch size tuning** - ROCm may need different optimal batch sizes vs CUDA
4. **Gradient accumulation** - More stable than large batch sizes on ROCm
5. **`paged_adamw_8bit`** - Works well on ROCm for memory savings

## Known Limitations

- Some CUDA-specific libraries may not have ROCm builds
- Flash Attention support is still maturing
- Multi-GPU (DDP) may need tuning for optimal performance
- Some quantization methods may have compatibility issues

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/#linux-rocm)
- [Hugging Face ROCm](https://huggingface.co/docs/optimum/amd)
- [bitsandbytes ROCm](https://github.com/ROCm/bitsandbytes)
