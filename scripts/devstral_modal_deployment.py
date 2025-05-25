"""
Custom Modal Labs deployment for Devstral:24b model optimized for tool calling
and development workflows in VS Code servers and agent setups.
"""

import os
import subprocess
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration Constants
MODEL_NAME: str = "devstral:24b"
CUSTOM_MODEL_NAME: str = "devstral-dev:24b"
OLLAMA_API_URL: str = "http://localhost:11434/api/version"
OLLAMA_SERVICE_NAME: str = "ollama"
OLLAMA_TIMEOUT: int = int(os.environ.get("OLLAMA_TIMEOUT", 60))
OLLAMA_POLL_INTERVAL: int = int(os.environ.get("OLLAMA_POLL_INTERVAL", 3))

# Model configuration for development workflows
DEV_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_ctx": 8192,
    "num_predict": 2048,
    "stop": ["<|endoftext|>", "</s>", "<|im_end|>"]
}

def run_subprocess(cmd: List[str], check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run subprocess with enhanced error handling and logging."""
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            capture_output=capture_output, 
            text=True,
            timeout=300
        )
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command '{' '.join(cmd)}' failed with code {e.returncode}: "
            f"stdout: {e.stdout}, stderr: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Command '{' '.join(cmd)}' timed out after 300s") from e

def setup_ollama_service() -> None:
    """Initialize and configure Ollama service."""
    commands = [
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", OLLAMA_SERVICE_NAME],
        ["systemctl", "start", OLLAMA_SERVICE_NAME]
    ]
    
    for cmd in commands:
        run_subprocess(cmd)

def wait_for_ollama_ready(timeout: int = OLLAMA_TIMEOUT) -> None:
    """Wait for Ollama service to be fully operational."""
    import httpx
    from loguru import logger
    
    start_time = time.time()
    while True:
        try:
            response = httpx.get(OLLAMA_API_URL, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Ollama service is ready")
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Ollama service failed to start within {timeout}s")
        
        logger.info(f"⏳ Waiting for Ollama service... ({int(elapsed)}s)")
        time.sleep(OLLAMA_POLL_INTERVAL)

def create_custom_model() -> None:
    """Create custom Devstral model with development-optimized configuration."""
    from loguru import logger
    
    # First, pull the base model
    logger.info(f"📥 Pulling base model: {MODEL_NAME}")
    run_subprocess(["ollama", "pull", MODEL_NAME])
    
    # Create custom model from Modelfile
    logger.info(f"🔧 Creating custom model: {CUSTOM_MODEL_NAME}")
    run_subprocess(["ollama", "create", CUSTOM_MODEL_NAME, "-f", "/app/Modelfile"])
    
    # Verify model creation
    result = run_subprocess(["ollama", "list"])
    if CUSTOM_MODEL_NAME not in result.stdout:
        raise RuntimeError(f"Failed to create custom model {CUSTOM_MODEL_NAME}")
    
    logger.info(f"✅ Custom model {CUSTOM_MODEL_NAME} created successfully")

# Enhanced Modal image with development tools
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "curl", "systemd", "git", "vim", "htop", 
        "build-essential", "software-properties-common"
    )
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
        "systemctl enable ollama"
    )
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "httpx==0.25.2",
        "loguru==0.7.2",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "aiofiles==23.2.1"
    )
    .copy_local_file("Modelfile", "/app/Modelfile")
)

# FastAPI application setup
api = FastAPI(
    title="Devstral Development Assistant API",
    description="Optimized Devstral:24b model for tool calling and development workflows",
    version="1.0.0"
)

# CORS middleware for VS Code server integration
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ToolCall(BaseModel):
    id: str = Field(..., description="Unique identifier for the tool call")
    type: str = Field(default="function", description="Type of tool call")
    function: Dict[str, Any] = Field(..., description="Function details")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, assistant, or tool")
    content: Optional[str] = Field(None, description="Message content")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls made by assistant")
    tool_call_id: Optional[str] = Field(None, description="ID of tool call being responded to")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default=CUSTOM_MODEL_NAME, description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Available tools")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field("auto", description="Tool selection strategy")
    stream: bool = Field(default=False, description="Stream response")
    max_tokens: Optional[int] = Field(2048, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.1, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter")

class HealthResponse(BaseModel):
    status: str
    model: str
    ready: bool
    timestamp: float

# API Endpoints
@api.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            models = response.json()
            model_available = any(
                model["name"].startswith(CUSTOM_MODEL_NAME) 
                for model in models.get("models", [])
            )
    except Exception:
        model_available = False
    
    return HealthResponse(
        status="healthy" if model_available else "degraded",
        model=CUSTOM_MODEL_NAME,
        ready=model_available,
        timestamp=time.time()
    )

@api.get("/v1/models")
async def list_models():
    """List available models - OpenAI API compatible."""
    return {
        "object": "list",
        "data": [
            {
                "id": CUSTOM_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "devstral-modal"
            }
        ]
    }

@api.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Enhanced chat completions with tool calling support."""
    import json
    import httpx
    from loguru import logger
    
    if not request.messages:
        raise HTTPException(400, "Messages array cannot be empty")
    
    # Prepare Ollama payload with development optimizations
    ollama_payload = {
        "model": request.model,
        "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
        "stream": request.stream,
        "options": {
            **DEV_CONFIG,
            "temperature": request.temperature or DEV_CONFIG["temperature"],
            "top_p": request.top_p or DEV_CONFIG["top_p"],
            "num_predict": request.max_tokens or DEV_CONFIG["num_predict"]
        }
    }
    
    # Add tools if provided
    if request.tools:
        ollama_payload["tools"] = request.tools
        ollama_payload["tool_choice"] = request.tool_choice
    
    ollama_url = "http://localhost:11434/api/chat"
    timestamp = int(time.time())
    
    def build_response(content: str, tool_calls: Optional[List] = None) -> dict:
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
            
        return {
            "id": f"chatcmpl-dev-{timestamp}",
            "object": "chat.completion",
            "created": timestamp,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }
    
    if request.stream:
        async def generate_stream() -> AsyncGenerator[str, None]:
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("POST", ollama_url, json=ollama_payload) as response:
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if "message" in data and "content" in data["message"]:
                                        chunk = {
                                            "id": f"chatcmpl-dev-{timestamp}",
                                            "object": "chat.completion.chunk",
                                            "created": timestamp,
                                            "model": request.model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": data["message"]["content"]},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk)}\n\n"
                                except json.JSONDecodeError:
                                    continue
                        
                        # Send final chunk
                        final_chunk = {
                            "id": f"chatcmpl-dev-{timestamp}",
                            "object": "chat.completion.chunk", 
                            "created": timestamp,
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_chunk = {"error": f"Stream error: {str(e)}"}
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    # Non-streaming response
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(ollama_url, json=ollama_payload)
            response.raise_for_status()
            data = response.json()
            
            content = data.get("message", {}).get("content", "")
            tool_calls = data.get("message", {}).get("tool_calls")
            
            return build_response(content, tool_calls)
            
    except httpx.TimeoutException:
        raise HTTPException(408, "Request timeout - model may be processing complex request")
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(500, f"Error processing request: {str(e)}")

# Modal App
app = modal.App(name="devstral-development-assistant", image=image)

@app.cls(
    gpu=modal.gpu.A10G(count=1),  # Optimal for 24b model
    cpu=4.0,
    memory=32768,  # 32GB RAM for 24b model
    container_idle_timeout=300,  # 5 minutes
    timeout=3600,  # 1 hour max execution
)
class DevstralService:
    """Modal service class for Devstral development assistant."""
    
    @modal.build()
    def build(self):
        """Build-time setup."""
        setup_ollama_service()
    
    @modal.enter()
    def startup(self):
        """Container startup - pull and configure model."""
        from loguru import logger
        
        logger.info("🚀 Starting Devstral Development Assistant")
        setup_ollama_service()
        wait_for_ollama_ready()
        create_custom_model()
        logger.info("✅ Devstral service ready for development workflows")
    
    @modal.asgi_app()
    def serve(self):
        """Serve the FastAPI application."""
        return api

# Local development support
@app.local_entrypoint()
def main():
    """Local testing entrypoint."""
    print("🔧 Testing Devstral deployment locally...")
    
    # You can add local testing logic here
    service = DevstralService()
    print("✅ Local test completed")

if __name__ == "__main__":
    # For local development
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)