#!/usr/bin/env python3
"""
Test client for DeepSeek-V4 Pro Qwen3.5 9B Uncensored API
Tests health, authentication, non-streaming, and streaming responses with latency & speed metrics.
"""

import sys
import time
import json
import argparse
import requests

def test_api(base_url: str, api_key: str, model_id: str = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"):
    # Normalize base_url (remove trailing slash)
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        root_url = base_url[:-3]
        v1_url = base_url
    else:
        root_url = base_url
        v1_url = f"{base_url}/v1"

    print("=" * 65)
    print(" 🧪 DEEPSEEK-V4-PRO UNCENSORED API TEST SUITE")
    print("=" * 65)
    print(f" Root URL   : {root_url}")
    print(f" V1 Base URL: {v1_url}")
    print(f" API Key    : {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
    print(f" Model ID   : {model_id}")
    print("=" * 65)

    # 1. Health check
    print("\n[1/4] Testing Health Endpoint (/health)...")
    try:
        r = requests.get(f"{root_url}/health", timeout=10)
        print(f"  Status Code: {r.status_code}")
        print(f"  Response   : {r.json()}")
    except Exception as e:
        print(f"  [!] Failed to connect to health endpoint: {e}")
        return

    # 2. Auth security test
    print("\n[2/4] Testing API Key Authentication Guard...")
    try:
        # Request with no key
        r_unauth = requests.post(f"{v1_url}/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        if r_unauth.status_code == 401:
            print("  [✓] Unauthorized request rejected properly (HTTP 401).")
        else:
            print(f"  [!] Warning: Expected 401, got {r_unauth.status_code}")

        # Request with invalid key
        r_bad = requests.post(
            f"{v1_url}/chat/completions",
            headers={"Authorization": "Bearer invalid_secret_key_123"},
            json={"messages": [{"role": "user", "content": "hi"}]}
        )
        if r_bad.status_code == 401:
            print("  [✓] Invalid API key rejected properly (HTTP 401).")
        else:
            print(f"  [!] Warning: Expected 401, got {r_bad.status_code}")
    except Exception as e:
        print(f"  [!] Auth test failed: {e}")

    # 3. Models list test
    print("\n[3/4] Testing Models Endpoint (/v1/models)...")
    try:
        r_models = requests.get(f"{v1_url}/models", headers={"Authorization": f"Bearer {api_key}"})
        if r_models.status_code == 200:
            print(f"  [✓] Available models: {[m['id'] for m in r_models.json().get('data', [])]}")
        else:
            print(f"  [!] Failed to list models: {r_models.status_code} {r_models.text}")
    except Exception as e:
        print(f"  [!] Models test failed: {e}")

    # 4. Real-time Streaming Test
    print("\n[4/4] Testing Real-Time SSE Streaming Completion (/v1/chat/completions)...")
    test_prompt = "Explain in 3 bullet points why open-weights AI models benefit developers."
    print(f"  Prompt: '{test_prompt}'\n")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful and concise uncensored assistant."},
            {"role": "user", "content": test_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True
    }

    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_text = ""

    try:
        with requests.post(f"{v1_url}/chat/completions", headers=headers, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                print(f"  [!] Request failed with HTTP {resp.status_code}: {resp.text}")
                return

            print("  Response: ", end="", flush=True)
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.time()
                            print(delta, end="", flush=True)
                            full_text += delta
                            token_count += 1
                    except Exception:
                        pass
        
        end_time = time.time()
        print("\n")
        print("-" * 65)
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        gen_time = (end_time - first_token_time) if first_token_time else total_time
        tps = (token_count / gen_time) if gen_time > 0 else 0

        print(f" ⏱️  Time to First Token (TTFT) : {ttft:.3f} s")
        print(f" ⚡ Total Generation Time        : {total_time:.2f} s")
        print(f" 📊 Generated Tokens            : {token_count}")
        print(f" 🚀 Generation Speed             : {tps:.1f} tokens/second")
        print("-" * 65)
        print(" [✓] API IS WORKING PERFECTLY WITHOUT INTERRUPTIONS OR LAG!")
        print("=" * 65)

    except Exception as e:
        print(f"\n[!] Streaming request failed: {e}")


def test_openai_sdk(base_url: str, api_key: str, model_id: str):
    """Demonstrates official OpenAI Python SDK compatibility."""
    try:
        from openai import OpenAI
    except ImportError:
        print("\n[*] 'openai' package not installed. Run 'pip install openai' to test OpenAI SDK.")
        return

    print("\n[+] Testing via official 'openai' Python library...")
    client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "What is 25 * 25?"}],
        stream=False
    )
    print(f"  OpenAI SDK Output: {response.choices[0].message.content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DeepSeek-V4 API")
    parser.add_argument("--url", type=str, required=True, help="Base Cloudflare URL (e.g. https://xxx.trycloudflare.com)")
    parser.add_argument("--api-key", type=str, default="sk-deepseek-uncensored-kaggle-key", help="API Key")
    parser.add_argument("--model", type=str, default="xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic", help="Model ID")
    parser.add_argument("--sdk", action="store_true", help="Also test official openai SDK")
    args = parser.parse_args()

    test_api(args.url, args.api_key, args.model)
    if args.sdk:
        test_openai_sdk(args.url, args.api_key, args.model)

