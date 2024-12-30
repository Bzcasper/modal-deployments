"""FastAPI endpoint that provides a simple pass-through to Ollama API.

This module provides a FastAPI application that forwards requests directly
to the Ollama API running within the container.
"""

import modal
import subprocess
import time
from fastapi import FastAPI, Request, HTTPException
from typing import Any
import httpx
from loguru import logger
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json


MODEL = "gemma2:27b"


def pull() -> None:
    """Initialize and pull the Ollama model."""
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    wait_for_ollama()
    subprocess.run(["ollama", "pull", MODEL], stdout=subprocess.PIPE)


def wait_for_ollama(timeout: int = 30, interval: int = 2) -> None:
    """Wait for Ollama service to be ready.

    :param timeout: Maximum time to wait in seconds
    :param interval: Time between checks in seconds
    :raises TimeoutError: If the service doesn't start within the timeout period
    """
    start_time = time.time()
    while True:
        try:
            response = httpx.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                logger.info("Ollama service is ready")
                return
        except httpx.ConnectError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Ollama service failed to start")
            logger.info(
                f"Waiting for Ollama service... ({int(time.time() - start_time)}s)"
            )
            time.sleep(interval)


# Configure Modal image with Ollama dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands(
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install("httpx", "loguru")
    .run_function(pull)
    .env(
        {
            "OLLAMA_HOST": "0.0.0.0",
            "OLLAMA_PORT": "11434",
        }
    )
)

app = modal.App(name="ollama", image=image)
api = FastAPI()


@api.get("/{path:path}")
@api.post("/{path:path}")
async def forward_request(path: str, request: Request) -> Any:
    """Forward incoming request to Ollama API.

    :param path: The path component of the request URL
    :param request: Incoming FastAPI request
    :return: Response from Ollama API or StreamingResponse for streaming endpoints
    """
    logger.info(f"Received {request.method} request for path: {path}")

    # First check if Ollama is running
    try:
        health_check = await httpx.AsyncClient().get(
            "http://localhost:11434/api/version"
        )
        logger.info(f"Ollama health check: {health_check.status_code}")
    except Exception as e:
        logger.error(f"Ollama health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Ollama service is not available")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if request.method == "GET":
                target_url = f"http://localhost:11434/{path}"
                logger.info(f"Forwarding GET request to: {target_url}")

                response = await client.get(target_url)
                logger.info(f"Received response with status: {response.status_code}")

                # Check if response is JSON before trying to parse it
                content_type = response.headers.get("content-type", "")
                logger.debug(f"Response content type: {content_type}")
                logger.debug(f"Response text: {response.text}")

                if "application/json" in content_type:
                    return response.json()
                return {"raw_response": response.text}

            else:  # POST
                body = await request.json()
                logger.info(f"Request body: {body}")

                # Check if this is a streaming endpoint
                if path == "api/chat" and body.get("stream", False):
                    logger.info("Handling streaming chat request")

                    async def generate() -> AsyncGenerator[bytes, None]:
                        async with httpx.AsyncClient() as stream_client:
                            async with stream_client.stream(
                                "POST",
                                f"http://localhost:11434/{path}",
                                json=body,
                                timeout=None,
                            ) as response:
                                async for line in response.aiter_lines():
                                    if line:  # Skip empty lines
                                        yield f"{line}\n"

                    return StreamingResponse(generate(), media_type="text/event-stream")

                # Non-streaming response
                target_url = f"http://localhost:11434/{path}"
                logger.info(f"Forwarding POST request to: {target_url}")

                response = await client.post(
                    target_url,
                    json=body,
                )
                logger.info(f"Received response with status: {response.status_code}")
                logger.debug(f"Response text: {response.text}")

                return response.json()

        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            logger.exception("Full traceback:")
            raise HTTPException(
                status_code=503, detail=f"Error forwarding request to Ollama: {str(e)}"
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Raw response text: {response.text}")
            return {"raw_response": response.text}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.exception("Full traceback:")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.cls(
    gpu=modal.gpu.A10G(count=1),
    container_idle_timeout=10,
)
class Ollama:
    """Modal container class for running Ollama service."""

    def __init__(self):
        """Initialize the Ollama service."""
        self.serve()

    @modal.build()
    def build(self):
        """Build step for Modal container setup."""
        subprocess.run(["systemctl", "daemon-reload"])
        subprocess.run(["systemctl", "enable", "ollama"])

    @modal.enter()
    def enter(self):
        """Entry point for Modal container."""
        subprocess.run(["systemctl", "start", "ollama"])
        wait_for_ollama()
        subprocess.run(["ollama", "pull", MODEL])

    @modal.asgi_app()
    def serve(self):
        """Serve the FastAPI application.

        :return: FastAPI application instance
        """
        return api
