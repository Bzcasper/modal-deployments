"""FastAPI endpoint for Ollama chat completions (OpenAI-compatible API)."""

import os
import subprocess
import time
from typing import Any, AsyncGenerator, List, Optional

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Constants and configuration
MODEL: str = os.environ.get("MODEL", "gemma2:27b")
DEFAULT_MODELS: List[str] = ["gemma2:27b"]
OLLAMA_API_URL: str = "http://localhost:11434/api/version"
OLLAMA_SERVICE_NAME: str = "ollama"
OLLAMA_PULL_TIMEOUT: int = int(os.environ.get("OLLAMA_PULL_TIMEOUT", 30))
OLLAMA_POLL_INTERVAL: int = int(os.environ.get("OLLAMA_POLL_INTERVAL", 2))


def run_subprocess(cmd: List[str], check: bool = True) -> None:
    """Run a subprocess command with error handling."""
    try:
        subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command '{' '.join(cmd)}' failed: {e.stderr.decode().strip()}"
        ) from e


def pull_model() -> None:
    """Initialize and pull the Ollama model."""
    run_subprocess(["systemctl", "daemon-reload"])
    run_subprocess(["systemctl", "enable", OLLAMA_SERVICE_NAME])
    run_subprocess(["systemctl", "start", OLLAMA_SERVICE_NAME])
    wait_for_ollama()
    run_subprocess(["ollama", "pull", MODEL])


def wait_for_ollama(
    timeout: int = OLLAMA_PULL_TIMEOUT, interval: int = OLLAMA_POLL_INTERVAL
) -> None:
    """Wait for Ollama service to be ready."""
    import httpx  # Note: 'httpx' is required for HTTP calls to Ollama API. The Modal image pip_installs it.
    from loguru import logger

    start_time = time.time()
    while True:
        try:
            response = httpx.get(OLLAMA_API_URL, timeout=5)
            if response.status_code == 200:
                logger.info("Ollama service is ready")
                return
        except httpx.RequestError:
            pass
        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.error("Ollama service failed to start after %ds", timeout)
            raise TimeoutError("Ollama service failed to start")
        logger.info(f"Waiting for Ollama service... ({int(elapsed)}s)")
        time.sleep(interval)


# Modal image setup
image = modal.Image.from_dockerfile(
    ".", dockerfile_path="Modelfile"  # context is project root
).pip_install("fastapi", "uvicorn", "httpx", "loguru", "pydantic")
app = modal.App(name="ollama", image=image)
api = FastAPI()


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender (e.g. 'user', 'assistant')"
    )
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(
        default=MODEL, description="The model to use for completion"
    )
    messages: List[ChatMessage] = Field(
        ..., description="The messages to generate a completion for"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")


@api.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest) -> Any:
    import json

    import httpx

    if not request.messages:
        raise HTTPException(
            status_code=400, detail="Messages array is required and cannot be empty"
        )
    if not request.model:
        raise HTTPException(status_code=400, detail="Model is required")
    if request.model not in DEFAULT_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Model '{request.model}' is not supported"
        )

    now = int(time.time())

    def build_response(content: str) -> dict:
        return {
            "id": f"chat-{now}",
            "object": "chat.completion",
            "created": now,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    ollama_url = "http://localhost:11434/api/chat"
    payload = {
        "model": request.model,
        "messages": [msg.model_dump() for msg in request.messages],
        "stream": request.stream,
    }

    if request.stream:

        async def generate_stream() -> AsyncGenerator[str, None]:
            async with httpx.AsyncClient() as client:
                try:
                    async with client.stream(
                        "POST", ollama_url, json=payload, timeout=None
                    ) as response:
                        async for line in response.aiter_lines():
                            if line:
                                yield f"data: {line}\n\n"
                        yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return build_response(data["message"]["content"])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat completion: {str(e)}"
        )


@app.cls(
    gpu=modal.gpu.A10G(count=1),
    container_idle_timeout=10,
)
class Ollama:
    """Modal container class for running Ollama service."""

    def __init__(self):
        self.serve()

    @modal.build()
    def build(self):
        run_subprocess(["systemctl", "daemon-reload"])
        run_subprocess(["systemctl", "enable", OLLAMA_SERVICE_NAME])

    @modal.enter()
    def enter(self):
        run_subprocess(["systemctl", "start", OLLAMA_SERVICE_NAME])
        wait_for_ollama()
        run_subprocess(["ollama", "pull", MODEL])

    @modal.asgi_app()
    def serve(self):
        return api
