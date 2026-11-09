#!/usr/bin/env bash
set -e

echo "=================================================================="
echo " 🚀 Setting up DeepSeek-V4-Pro (Qwen3.5-9B) Uncensored on Kaggle"
echo "=================================================================="

# 1. Update pip and install python dependencies
echo "[*] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Install prebuilt llama-cpp-python with CUDA 12.x support or standalone llama-server
echo "[*] Setting up high-performance CUDA inference backend..."
# Attempt to install CUDA-accelerated wheel for llama-cpp-python
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
  --no-cache-dir || {
    echo "[!] Fallback: compiling llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
}

# 3. Download cloudflared for Linux x86_64
echo "[*] Setting up Cloudflare Tunnel..."
if ! command -v cloudflared &> /dev/null; then
    curl -L --output /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x /usr/local/bin/cloudflared
fi
echo "[✓] cloudflared is ready: $(cloudflared --version)"

# 4. Download Model GGUF weights
echo "[*] Downloading DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic GGUF (Q4_K_M, 5.2 GB)..."
python3 -c "
from huggingface_hub import hf_hub_download
import os

model_path = hf_hub_download(
    repo_id='mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF',
    filename='DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf',
    local_dir='./models',
    local_dir_use_symlinks=False
)
print(f'[✓] Model downloaded successfully to: {model_path}')
"

echo "=================================================================="
echo " [✓] Environment & Model Setup Completed!"
echo "=================================================================="

