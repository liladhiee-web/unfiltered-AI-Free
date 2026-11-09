#!/usr/bin/env python3
"""
Cloudflare Free Tunnel Launcher for Kaggle / Linux
Provides free, secure HTTPS public URL for local API servers without port forwarding.
"""

import os
import re
import sys
import stat
import time
import shutil
import urllib.request
import subprocess
import threading
from typing import Optional

CLOUDFLARED_URLS = {
    "linux": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "darwin": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64",
    "win32": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
}

TUNNEL_REGEX = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def get_binary_path() -> str:
    """Check if cloudflared is already installed or download it."""
    # Check if already installed in system PATH
    found = shutil.which("cloudflared")
    if found:
        return found

    # Local binary path
    ext = ".exe" if sys.platform == "win32" else ""
    local_bin = os.path.abspath(f"./cloudflared{ext}")
    if os.path.exists(local_bin):
        return local_bin

    # Download for platform
    url = CLOUDFLARED_URLS.get(sys.platform, CLOUDFLARED_URLS["linux"])
    print(f"[*] Downloading Cloudflare Tunnel binary from:\n    {url}")
    
    urllib.request.urlretrieve(url, local_bin)
    
    # Make executable on Unix
    if sys.platform != "win32":
        st = os.stat(local_bin)
        os.chmod(local_bin, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print(f"[✓] cloudflared downloaded to: {local_bin}")
    return local_bin


class CloudflareTunnel:
    def __init__(self, port: int = 8000):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.public_url: Optional[str] = None
        self._stop_event = threading.Event()

    def start(self, timeout: int = 45) -> str:
        """Starts cloudflared and blocks until the public URL is captured."""
        binary = get_binary_path()
        cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{self.port}", "--no-autoupdate"]

        print(f"[*] Starting Cloudflare Quick Tunnel on port {self.port}...")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    raise RuntimeError(f"cloudflared exited unexpectedly with code {self.process.returncode}")
                time.sleep(0.1)
                continue

            match = TUNNEL_REGEX.search(line)
            if match:
                self.public_url = match.group(0)
                with open("tunnel_url.txt", "w", encoding="utf-8") as f:
                    f.write(self.public_url)
                
                # Start background log consumer to keep pipes clear
                threading.Thread(target=self._drain_logs, daemon=True).start()
                return self.public_url

        raise TimeoutError("Failed to extract Cloudflare tunnel URL within timeout period.")

    def _drain_logs(self):
        """Continuously reads output so the pipe buffer never fills up."""
        try:
            while not self._stop_event.is_set() and self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    break
        except Exception:
            pass

    def stop(self):
        """Terminates the tunnel process."""
        self._stop_event.set()
        if self.process:
            print("[*] Stopping Cloudflare Tunnel...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


def print_banner(public_url: str, api_key: str, model_id: str):
    """Prints a beautiful, formatted console banner with credentials and quick-start commands."""
    divider = "═" * 70
    thin_div = "─" * 70
    print(f"\n{divider}")
    print(" 🚀 DEEPSEEK-V4-PRO (QWEN3.5-9B) UNCENSORED API IS LIVE ON KAGGLE GPU!")
    print(divider)
    print(f" 🌐 PUBLIC BASE URL : {public_url}/v1")
    print(f" 🖥️  WEB CHAT UI     : {public_url}")
    print(f" 🔑 API KEY         : {api_key}")
    print(f" 📦 MODEL ID        : {model_id}")
    print(thin_div)
    print(" 💡 PYTHON OPENAI SDK USAGE:")
    print("    from openai import OpenAI")
    print(f"    client = OpenAI(base_url='{public_url}/v1', api_key='{api_key}')")
    print(f"    res = client.chat.completions.create(")
    print(f"        model='{model_id}',")
    print("        messages=[{'role': 'user', 'content': 'Hello! Write a Python script.'}],")
    print("        stream=True")
    print("    )")
    print("    for chunk in res: print(chunk.choices[0].delta.content or '', end='', flush=True)")
    print(thin_div)
    print(" ⚡ CURL TEST COMMAND:")
    print(f"    curl -N -X POST '{public_url}/v1/chat/completions' \\")
    print(f"         -H 'Authorization: Bearer {api_key}' \\")
    print("         -H 'Content-Type: application/json' \\")
    print(f"         -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hi!\"}}], \"stream\": true}}'")
    print(f"{divider}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cloudflare Tunnel Helper")
    parser.add_argument("--port", type=int, default=8000, help="Local port to expose")
    parser.add_argument("--api-key", type=str, default=os.getenv("API_KEY", "sk-deepseek-uncensored-kaggle-key"), help="API Key for display")
    parser.add_argument("--model", type=str, default="xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic", help="Model ID")
    args = parser.parse_args()

    tunnel = CloudflareTunnel(port=args.port)
    try:
        url = tunnel.start()
        print_banner(url, args.api_key, args.model)
        print("[*] Tunnel running. Press Ctrl+C to terminate.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tunnel.stop()
        print("[*] Tunnel closed.")

