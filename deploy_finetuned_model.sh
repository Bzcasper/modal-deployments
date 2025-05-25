#!/bin/bash

echo "🚀 Deploying Fine-tuned Devstral Reasoning Model"

# Set environment variables
export MODAL_TOKEN_ID=${MODAL_TOKEN_ID}
export MODAL_TOKEN_SECRET=${MODAL_TOKEN_SECRET}
export HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}

# Deploy fine-tuning pipeline
echo "📦 Deploying fine-tuning pipeline to Modal..."
modal deploy ollama_finetuning.py

# Run fine-tuning
echo "🎯 Starting fine-tuning process..."
modal run ollama_finetuning.py::main --base-model="mistralai/Mistral-7B-v0.1"

# Convert and deploy to Ollama
echo "🔄 Converting model for Ollama..."
modal run ollama_integration.py::convert_to_ollama_format

echo "✅ Fine-tuned model deployment complete!"
echo "🔧 To use the model: ollama run devstral-reasoning:24b"