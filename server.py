#!/usr/bin/env python3
"""
High-Performance OpenAI-Compatible API Gateway for DeepSeek-V4 Pro Qwen3.5 9B Uncensored
Provides API key security, real-time SSE streaming, Web Chat UI, and OpenAI compatibility.
"""

import os
import sys
import time
import json
import uuid
import asyncio
from typing import List, Optional, Union, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends, Header, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# Configuration
DEFAULT_MODEL_ID = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")
API_KEY = os.getenv("API_KEY", "sk-deepseek-uncensored-kaggle-key")
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(
    title="DeepSeek-V4-Pro Uncensored API",
    version="1.0.0",
    description="High-speed, OpenAI-compatible API running on Kaggle Free GPU via Cloudflare Tunnel"
)

# Enable CORS for all frontends (OpenWebUI, SillyTavern, LibreChat, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas for OpenAI compatibility
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL_ID
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None

class CompletionRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL_ID
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


# Security / API Key Verification Dependency
def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    api_key_query: Optional[str] = Query(None, alias="api_key")
):
    """Verifies the API key via Bearer header, X-API-Key header, or ?api_key query parameter."""
    provided_key = None
    if authorization and authorization.lower().startswith("bearer "):
        provided_key = authorization[7:].strip()
    elif x_api_key:
        provided_key = x_api_key.strip()
    elif api_key_query:
        provided_key = api_key_query.strip()

    if not provided_key or provided_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key. Supply 'Authorization: Bearer <API_KEY>' or 'X-API-Key: <API_KEY>'."
        )
    return provided_key


# Web UI
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>DeepSeek-V4 Pro Qwen3.5 9B Uncensored API is Active</h2>")


# Health Check
@app.get("/health")
async def health_check():
    backend_status = "UNKNOWN"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{BACKEND_URL}/health")
            backend_status = "ONLINE" if resp.status_code == 200 else f"HTTP {resp.status_code}"
    except Exception as e:
        backend_status = f"CONNECT_FAIL ({e})"

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model": DEFAULT_MODEL_ID,
        "backend": BACKEND_URL,
        "backend_status": backend_status
    }


# OpenAI Models Endpoint
@app.get("/v1/models")
async def list_models(key: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xtools-at",
                "permission": [],
                "root": DEFAULT_MODEL_ID,
                "parent": None
            },
            {
                "id": "deepseek-v4-uncensored",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xtools-at",
                "permission": [],
                "root": DEFAULT_MODEL_ID,
                "parent": None
            }
        ]
    }


# OpenAI Chat Completions Endpoint
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, key: str = Depends(verify_api_key)):
    payload = req.model_dump()
    # Normalize model name for backend
    payload["model"] = DEFAULT_MODEL_ID

    if req.stream:
        return StreamingResponse(
            stream_backend_chat(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    f"{BACKEND_URL}/v1/chat/completions",
                    json=payload,
                    timeout=180.0
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Inference Backend Unreachable: {e}")


async def stream_backend_chat(payload: dict):
    """Streams SSE tokens directly from backend without delay or buffering."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            async with client.stream("POST", f"{BACKEND_URL}/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    err_msg = await response.aread()
                    yield f"data: {json.dumps({'error': err_msg.decode('utf-8')})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"


# OpenAI Text Completions Endpoint
@app.post("/v1/completions")
async def legacy_completions(req: CompletionRequest, key: str = Depends(verify_api_key)):
    payload = req.model_dump()
    payload["model"] = DEFAULT_MODEL_ID

    if req.stream:
        return StreamingResponse(
            stream_backend_legacy(payload),
            media_type="text/event-stream"
        )
    else:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    f"{BACKEND_URL}/v1/completions",
                    json=payload,
                    timeout=180.0
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Inference Backend Unreachable: {e}")


async def stream_backend_legacy(payload: dict):
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            async with client.stream("POST", f"{BACKEND_URL}/v1/completions", json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="API Gateway Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to listen on")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="Secret API Key")
    parser.add_argument("--backend-url", type=str, default=BACKEND_URL, help="Backend URL")
    args = parser.parse_args()

    API_KEY = args.api_key
    BACKEND_URL = args.backend_url
    print(f"[*] Starting API Gateway on port {args.port} pointing to backend {BACKEND_URL}")
    print(f"[*] API Key set to: {API_KEY}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

