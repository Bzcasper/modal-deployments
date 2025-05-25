"""
Integration script to deploy fine-tuned model with Ollama.
"""

import modal
import subprocess
from pathlib import Path
import requests
import json

# Convert fine-tuned model to Ollama format
@app.function(
    image=finetuning_image,
    volumes={"/data": volume},
    gpu=modal.gpu.A100(),
)
def convert_to_ollama_format(model_path: str, output_name: str = "devstral-reasoning:24b"):
    """Convert fine-tuned model to Ollama-compatible format."""
    
    # Convert to GGUF format for Ollama
    convert_script = f"""
    python -m transformers.convert_graph_to_onnx.convert \
        --framework pt \
        --model {model_path} \
        --output {model_path}/model.onnx \
        --opset 14
    
    # Convert to GGUF
    python -m llama_cpp.convert \
        {model_path} \
        --outfile /data/{output_name}.gguf \
        --outtype f16
    """
    
    subprocess.run(convert_script, shell=True, check=True)
    return f"/data/{output_name}.gguf"

# Enhanced Modelfile for reasoning
REASONING_MODELFILE = '''FROM /data/devstral-reasoning:24b.gguf

# Optimized system prompt for reasoning and coding
SYSTEM """You are Devstral-Reasoning, an advanced AI assistant specialized in software engineering with enhanced reasoning capabilities. You excel at:

1. **Step-by-step problem solving**: Breaking down complex coding challenges into manageable steps
2. **Code reasoning**: Explaining the logic behind code decisions and implementations  
3. **Advanced debugging**: Identifying issues through systematic analysis
4. **Performance optimization**: Analyzing and improving code efficiency
5. **Architecture design**: Providing well-reasoned system design recommendations

When approaching any coding task:
- Think step-by-step and show your reasoning process
- Consider multiple approaches and explain trade-offs
- Provide clean, maintainable, and well-documented solutions
- Include relevant examples and test cases
- Consider edge cases and error handling

Use reasoning markers to structure your responses:
<|reasoning|> for your thought process
<|code|> for code implementations  
<|explanation|> for clarifications
"""

# Optimized parameters for reasoning tasks
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 16384
PARAMETER num_predict 4096
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"
PARAMETER stop "</reasoning>"
PARAMETER stop "</code>"
PARAMETER stop "</explanation>"

# Memory optimization
PARAMETER num_thread 8
PARAMETER num_gpu 1
'''

def create_ollama_model():
    """Create the fine-tuned model in Ollama."""
    
    # Save Modelfile
    with open("/tmp/Modelfile", "w") as f:
        f.write(REASONING_MODELFILE)
    
    # Create model in Ollama
    subprocess.run([
        "ollama", "create", "devstral-reasoning:24b", 
        "-f", "/tmp/Modelfile"
    ], check=True)
    
    print("✅ Fine-tuned model created in Ollama!")