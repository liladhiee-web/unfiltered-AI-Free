#!/usr/bin/env bash
set -e

echo "=================================================================="
echo " 🚀 Setting up DeepSeek-V4-Pro (Qwen3.5-9B) Uncensored on Kaggle"
echo "=================================================================="

# 1. Update pip and install python dependencies & CUDA runtimes
echo "[*] Installing Python dependencies & CUDA 12 runtime packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Register CUDA 12 libraries with system dynamic linker (resolves libcudart.so.12)
echo "[*] Configuring system dynamic linker for CUDA 12..."
python3 -c "
import os, glob, site

cuda_dirs = set()
for p in ['/usr/local/cuda/lib64', '/usr/local/cuda-12/lib64', '/usr/local/cuda-*/lib64', '/usr/lib/x86_64-linux-gnu']:
    for m in glob.glob(p):
        if os.path.isdir(m): cuda_dirs.add(m)

for sp in site.getsitepackages():
    for sub in ['nvidia/cuda_runtime/lib', 'nvidia/cublas/lib']:
        d = os.path.join(sp, sub)
        if os.path.isdir(d): cuda_dirs.add(d)

conf_path = '/etc/ld.so.conf.d/cuda_custom.conf'
try:
    with open(conf_path, 'w') as f:
        f.write('\n'.join(cuda_dirs) + '\n')
    os.system('ldconfig 2>/dev/null')
    print(f'[✓] Registered {len(cuda_dirs)} CUDA library directories with ldconfig')
except Exception as e:
    print(f'[!] ldconfig note: {e}')
"

# 3. Install prebuilt llama-cpp-python with CUDA 12.x support
echo "[*] Setting up high-performance CUDA inference backend..."
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
  --no-cache-dir -q || {
    echo "[!] Compiling llama-cpp-python from source with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir -q
}

# 4. Download cloudflared for Linux x86_64
echo "[*] Setting up Cloudflare Tunnel..."
if ! command -v cloudflared &> /dev/null; then
    curl -L --output /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x /usr/local/bin/cloudflared
fi
echo "[✓] cloudflared is ready: $(cloudflared --version)"

# 5. Download Model GGUF weights
echo "[*] Downloading DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic GGUF (Q4_K_M, 5.2 GB)..."
python3 -c "
import os
from huggingface_hub import hf_hub_download

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
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
