#!/bin/bash
# IndoROC: ROCm Environment Setup Script
# For AMD Developer Cloud (Instinct MI250/MI300)

set -e

echo "🇮🇩 IndoROC: Setting up ROCm environment on AMD Developer Cloud"
echo "================================================================"

# Check if running on AMD GPU instance
echo ""
echo "🔍 Checking GPU availability..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
    echo "✅ ROCm detected"
else
    echo "⚠️  rocm-smi not found. Make sure you're on an AMD GPU instance."
fi

# System packages
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
    git \
    wget \
    curl \
    python3.10 \
    python3.10-venv \
    python3-pip \
    build-essential \
    libsndfile1 \
    ffmpeg

# Create virtual environment
echo ""
echo "🐍 Setting up Python virtual environment..."
python3.10 -m venv ~/indo-rocm-venv
source ~/indo-rocm-venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with ROCm
echo ""
echo "🔥 Installing PyTorch with ROCm backend..."
# ROCm 6.0+ compatible PyTorch
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0

# Verify PyTorch + GPU
echo ""
echo "🔍 Verifying PyTorch + GPU..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'ROCm version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')
    # Quick test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x.T)
    print(f'✅ GPU compute test passed! (matrix 1000x1000)')
else:
    print('❌ GPU not available!')
"

# Install Hugging Face ecosystem
echo ""
echo "🤗 Installing Hugging Face ecosystem..."
pip install \
    transformers>=4.40.0 \
    datasets>=2.19.0 \
    accelerate>=0.30.0 \
    peft>=0.11.0 \
    bitsandbytes>=0.43.0 \
    trl>=0.8.0 \
    sentencepiece \
    protobuf \
    safetensors

# Install vLLM for inference (ROCm build)
echo ""
echo "⚡ Installing vLLM (ROCm build)..."
pip install vllm || echo "⚠️  vLLM ROCm build may need manual installation"

# Install additional tools
echo ""
echo "🔧 Installing additional tools..."
pip install \
    pyyaml \
    tensorboard \
    wandb \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    jupyter \
    ipywidgets

# Install vision pipeline dependencies
echo ""
echo "👁️  Installing vision pipeline dependencies..."
pip install \
    opencv-python-headless \
    pillow \
    paddlepaddle-gpu \
    paddleocr \
    fastapi \
    uvicorn \
    python-multipart

# Environment variables
echo ""
echo "⚙️  Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# IndoROC: ROCm Environment Variables
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# PyTorch ROCm optimizations
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Adjust for your GPU

# Disable unnecessary features
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
EOF

source ~/.bashrc

echo ""
echo "================================================================"
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate venv: source ~/indo-rocm-venv/bin/activate"
echo "   2. Prepare data:  python src/indo-llm/prepare_data.py"
echo "   3. Train model:   python src/indo-llm/train.py --config configs/lora_config.yaml"
echo ""
echo "💡 Tips:"
echo "   - Monitor GPU: watch -n 1 rocm-smi"
echo "   - Check VRAM: python -c 'import torch; print(torch.cuda.memory_summary())'"
echo "================================================================"
