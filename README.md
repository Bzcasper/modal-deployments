# Modal Deployments

This repository contains Modal deployments for running AI services in the cloud. Currently, it includes two main services:

## 1. Ollama API Service (`ollama_api.py`)

An OpenAI-compatible API interface for Ollama models, built with FastAPI and Modal. This service allows you to interact with Ollama models using the familiar OpenAI API format.

### Features
- OpenAI-compatible chat completions endpoint (`/v1/chat/completions`)
- Supports both streaming and non-streaming responses
- Runs on Modal with GPU acceleration (A10G)
- Default model: gemma2:27b (configurable via environment variables)

### Usage

```python
import openai

# Configure the client to use your Modal endpoint
client = openai.Client(
    base_url="https://YOUR-MODAL-DEPLOYMENT.modal.run/v1",
    api_key="not-needed"  # The API doesn't require authentication
)

# Make a chat completion request
response = client.chat.completions.create(
    model="gemma2:27b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=False  # Set to True for streaming responses
)
```

## 2. Nougat OCR Service (`nougat_ocr_function.py`)

A Modal function deployment for running Nougat OCR on PDF files. This service extracts text from academic papers and technical documents while preserving their formatting.

### Features

- GPU-accelerated OCR processing (A10G)
- Markdown-compatible output
- Support for processing specific page ranges
- Uses Nougat base model (0.1.0-base) for high-quality extraction

### Usage

```python
import modal

# Look up the deployed function
f = modal.Function.lookup("nougat-ocr-cli", "process_pdf")

# Process a PDF file
with open("paper.pdf", "rb") as file:
    pdf_bytes = file.read()
text = f.remote(pdf_bytes)

# Process specific pages
text = f.remote(pdf_bytes, start_page=1, end_page=5)
```

## Deployment

To deploy either service to Modal:

```bash
# Deploy Ollama API
modal deploy deployments/ollama_api.py

# Deploy Nougat OCR
modal deploy deployments/nougat_ocr_function.py
```

## Requirements

- Modal account and CLI setup
- For local development: Python 3.12+
- GPU requirements are handled automatically by Modal

## Environment Variables

- `MODEL`: (Ollama API) Specifies which model to use (default: "gemma2:27b")

## Architecture

Both services use Modal's GPU-enabled containers and are designed for scalability and ease of deployment. The Ollama API service runs as a persistent FastAPI application, while the Nougat OCR service operates as a serverless function.
