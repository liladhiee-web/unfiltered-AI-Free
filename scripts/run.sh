#!/usr/bin/env bash
set -e

API_KEY="${API_KEY:-sk-deepseek-uncensored-kaggle-key}"
PORT=8000
BACKEND_PORT=8080
MODEL_PATH="./models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"

echo "=================================================================="
echo " 🚀 Launching DeepSeek-V4-Pro 9B Uncensored API Stack"
echo "=================================================================="
echo " Model      : $MODEL_PATH"
echo " API Key    : $API_KEY"
echo " Main Port  : $PORT"
echo "=================================================================="

# 0. Ensure CUDA 12 library paths are in LD_LIBRARY_PATH
python3 -c "
import os, glob, site

dirs = []
for p in ['/usr/local/cuda/lib64', '/usr/local/cuda-12/lib64', '/usr/local/cuda-*/lib64']:
    for m in glob.glob(p):
        if os.path.isdir(m) and m not in dirs: dirs.append(m)

for sp in site.getsitepackages():
    for sub in ['nvidia/cuda_runtime/lib', 'nvidia/cublas/lib']:
        d = os.path.join(sp, sub)
        if os.path.isdir(d) and d not in dirs: dirs.append(d)

print(':'.join(dirs))
" > /tmp/cuda_ld_dirs.txt

CUDA_EXTRA_LD=$(cat /tmp/cuda_ld_dirs.txt)
export LD_LIBRARY_PATH="${CUDA_EXTRA_LD}:${LD_LIBRARY_PATH}"
echo "[*] Configured LD_LIBRARY_PATH for CUDA execution."

# 1. Start CUDA inference engine in background (all 99 layers offloaded to GPU)
echo "[*] Launching high-speed CUDA backend on port $BACKEND_PORT..."
python3 -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --n_gpu_layers 99 \
    --n_ctx 8192 \
    --n_batch 512 \
    --host 127.0.0.1 \
    --port $BACKEND_PORT \
    --chat_format chatml > backend.log 2>&1 &
BACKEND_PID=$!
echo "[✓] Backend started with PID: $BACKEND_PID"

# Wait for backend to be ready
echo "[*] Waiting for model to load into GPU VRAM..."
for i in {1..60}; do
    if curl -s "http://127.0.0.1:$BACKEND_PORT/docs" > /dev/null; then
        echo "[✓] CUDA inference backend is ONLINE and ready!"
        break
    fi
    sleep 2
done

# 2. Start API Gateway on port 8000
echo "[*] Starting FastAPI Gateway..."
API_KEY="$API_KEY" BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" python3 server.py --port $PORT --api-key "$API_KEY" > gateway.log 2>&1 &
GATEWAY_PID=$!
echo "[✓] Gateway started with PID: $GATEWAY_PID"

# Wait 2 seconds for gateway
sleep 2

# 3. Start Keep-Alive daemon
python3 keep_alive.py --port $PORT > keep_alive.log 2>&1 &
KEEPALIVE_PID=$!
echo "[✓] Keep-Alive daemon started with PID: $KEEPALIVE_PID"

# 4. Start Cloudflare Tunnel and display public URL
echo "[*] Establishing Cloudflare Secure Tunnel..."
python3 tunnel.py --port $PORT --api-key "$API_KEY"
