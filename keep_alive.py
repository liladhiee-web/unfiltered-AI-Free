#!/usr/bin/env python3
"""
Anti-Idle / Keep-Alive Daemon for Kaggle GPU Sessions
Ensures long-running inference jobs stay active and log vital GPU statistics.
"""

import os
import sys
import time
import urllib.request
import subprocess
import threading
from datetime import datetime


def get_gpu_status() -> str:
    """Queries nvidia-smi for concise VRAM and GPU utilization info."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            stats = []
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    stats.append(f"GPU {i} ({parts[0]}): {parts[1]}/{parts[2]} MB VRAM | Util: {parts[3]}% | Temp: {parts[4]}°C")
            return " | ".join(stats)
    except Exception:
        pass
    return "GPU info unavailable"


def run_keep_alive(port: int = 8000, interval_seconds: int = 60, stop_event: threading.Event = None):
    """Periodically heartbeats the local API server and logs memory stats."""
    print(f"[*] Keep-Alive service active (Heartbeat every {interval_seconds}s)")
    start_time = time.time()
    
    while stop_event is None or not stop_event.is_set():
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        uptime_min = (time.time() - start_time) / 60
        gpu_stat = get_gpu_status()

        # Ping health endpoint
        status_msg = "HEALTHY"
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"User-Agent": "KeepAlivePing"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    status_msg = f"HTTP {resp.status}"
        except Exception as e:
            status_msg = f"Ping Failed: {e}"

        print(f"[{now_str}] 🟢 Heartbeat [Uptime: {uptime_min:.1f}m] | API: {status_msg} | {gpu_stat}", flush=True)

        # Sleep in small slices to respond quickly to stop event
        for _ in range(interval_seconds):
            if stop_event and stop_event.is_set():
                break
            time.sleep(1)


def start_keep_alive_thread(port: int = 8000, interval: int = 60) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    t = threading.Thread(target=run_keep_alive, args=(port, interval, stop_event), daemon=True)
    t.start()
    return t, stop_event


if __name__ == "__main__":
    try:
        run_keep_alive()
    except KeyboardInterrupt:
        print("[*] Keep-alive terminated.")

