import modal
import os
import subprocess
import time
from loguru import logger

# Configuration
MODEL = os.environ.get("MODEL", "llama3.2-vision")
DEFAULT_MODELS = ["llama3.2-vision"]


def wait_for_ollama(timeout: int = 30, interval: int = 2) -> None:
    """Wait for Ollama service to be ready."""
    import httpx

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


def setup_ollama() -> None:
    """Initialize and setup the Ollama service."""
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    wait_for_ollama()
    subprocess.run(["ollama", "pull", MODEL], stdout=subprocess.PIPE)


# Define the Modal image
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
    .pip_install("ollama", "httpx", "loguru")
    .run_function(setup_ollama)
    .env(
        {
            "OLLAMA_HOST": "0.0.0.0",
            "OLLAMA_PORT": "11434",
        }
    )
)

# Create the Modal app
app = modal.App("ollama-app")


@app.function(image=image, gpu="A10G", timeout=10)
@modal.web_endpoint(method="GET")
def start_ollama():
    """Start Ollama service and expose it through a tunnel."""
    # Start Ollama service
    subprocess.run(["systemctl", "start", "ollama"])
    wait_for_ollama()

    # Forward the Ollama port through a tunnel
    with modal.forward(11434) as tunnel:
        print(f"tunnel.url        = {tunnel.url}")
        print(f"tunnel.tls_socket = {tunnel.tls_socket}")
        # logger.info(f"Ollama is accessible at: {tunnel.url}")

        # Keep the service running
        while True:
            time.sleep(1)
