"""
Advanced fine-tuning implementation for Ollama models with Modal Labs.
Incorporates reasoning datasets and state-of-the-art techniques.
"""

import modal
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb
from loguru import logger

# Modal configuration for fine-tuning
finetuning_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "accelerate>=0.23.0",
        "bitsandbytes>=0.41.0",
        "wandb>=0.15.0",
        "loguru>=0.7.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0"
    )
    .run_commands("huggingface-cli login --token $HUGGINGFACE_TOKEN || true")
)

app = modal.App("ollama-advanced-finetuning", image=finetuning_image)
volume = modal.Volume.from_name("ollama-finetuning-data", create_if_missing=True)

class AdvancedFineTuningConfig:
    """Advanced configuration for model fine-tuning."""
    
    # LoRA Configuration optimized for coding tasks
    LORA_CONFIG = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,  # Higher rank for coding tasks
        lora_alpha=128,  # 2x rank for stable training
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"]  # Save embedding layers
    )
    
    # Training arguments optimized for reasoning
    TRAINING_ARGS = TrainingArguments(
        output_dir="/data/checkpoints",
        overwrite_output_dir=True,
        
        # Learning rate and scheduling
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        
        # Batch size and gradient settings
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        
        # Training duration
        num_train_epochs=3,
        max_steps=-1,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps", 
        save_steps=500,
        save_total_limit=3,
        
        # Optimization
        optim="adamw_bnb_8bit",  # Memory efficient optimizer
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Mixed precision
        fp16=True,
        dataloader_pin_memory=True,
        
        # Logging
        logging_steps=10,
        report_to="wandb",
        run_name="ollama-coding-reasoning-finetune",
        
        # Memory optimization
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

@app.cls(
    gpu=modal.gpu.A100(count=2),  # Multi-GPU for faster training
    cpu=16.0,
    memory=64000,  # 64GB RAM
    volumes={"/data": volume},
    timeout=86400,  # 24 hours
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("wandb-token")
    ]
)
class AdvancedOllamaFineTuner:
    """Advanced fine-tuning class for Ollama models."""
    
    def __init__(self):
        self.config = AdvancedFineTuningConfig()
        self.model = None
        self.tokenizer = None
        
    @modal.enter()
    def setup(self):
        """Initialize the fine-tuning environment."""
        import os
        
        # Set up Weights & Biases
        if "WANDB_API_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_API_KEY"])
        
        logger.info("🚀 Fine-tuning environment initialized")
    
    @modal.method()
    async def prepare_model_and_tokenizer(self, base_model: str = "codellama/CodeLlama-7b-hf"):
        """Prepare base model and tokenizer for fine-tuning."""
        logger.info(f"📦 Loading base model: {base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add special tokens for reasoning
        special_tokens = {
            "additional_special_tokens": [
                "<|reasoning|>", "</reasoning>",
                "<|code|>", "</code>",
                "<|explanation|>", "</explanation>",
                "<|step|>", "</step>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True,  # 8-bit quantization
            cache_dir="/data/model_cache"
        )
        
        # Resize token embeddings for new special tokens
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA
        self.model = get_peft_model(model, self.config.LORA_CONFIG)
        self.model.print_trainable_parameters()
        
        logger.info("✅ Model and tokenizer prepared")
        return True
    
    @modal.method()
    async def create_reasoning_dataset(self) -> str:
        """Create and save the reasoning dataset."""
        from setup.dataset_preparation import AdvancedCodeDatasetBuilder
        
        builder = AdvancedCodeDatasetBuilder()
        
        logger.info("🔨 Building reasoning dataset...")
        reasoning_examples = await builder.create_reasoning_dataset()
        
        # Process examples for training
        processed_examples = []
        for example in reasoning_examples:
            # Format as conversation with reasoning
            conversation = self._format_reasoning_conversation(example)
            processed_examples.append(conversation)
        
        # Save dataset
        dataset_path = "/data/reasoning_dataset.json"
        async with aiofiles.open(dataset_path, 'w') as f:
            await f.write(json.dumps(processed_examples, indent=2))
        
        logger.info(f"💾 Dataset saved: {len(processed_examples)} examples")
        return dataset_path
    
    def _format_reasoning_conversation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Format example as a reasoning conversation."""
        
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert software engineer who thinks step-by-step and provides detailed reasoning for code solutions."
                },
                {
                    "role": "user", 
                    "content": f"{example['instruction']}\n\n```{example.get('language', 'python')}\n{example['input']}\n```"
                },
                {
                    "role": "assistant",
                    "content": example['output']
                }
            ]
        }
        
        return conversation
    
    @modal.method()
    async def fine_tune_model(self, dataset_path: str):
        """Execute the fine-tuning process."""
        logger.info("🎯 Starting fine-tuning process")
        
        # Load and prepare dataset
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        
        # Tokenize dataset
        train_dataset = self._tokenize_dataset(raw_data)
        
        # Split for validation
        train_size = int(0.95 * len(train_dataset))
        eval_dataset = Dataset.from_dict({
            k: v[train_size:] for k, v in train_dataset.data.items()
        })
        train_dataset = Dataset.from_dict({
            k: v[:train_size] for k, v in train_dataset.data.items()
        })
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.config.TRAINING_ARGS,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("🚂 Training started...")
        trainer.train()
        
        # Save final model
        final_model_path = "/data/final_model"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"✅ Fine-tuning completed! Model saved to {final_model_path}")
        return final_model_path
    
    def _tokenize_dataset(self, raw_data: List[Dict]) -> Dataset:
        """Tokenize the dataset for training."""
        
        def tokenize_conversation(examples):
            texts = []
            for conversation in examples["conversations"]:
                # Format conversation as single text
                text = self._conversation_to_text(conversation)
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )
            
            # Set labels for causal LM
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Convert to dataset format
        dataset_dict = {"conversations": raw_data}
        dataset = Dataset.from_dict(dataset_dict)
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_conversation,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def _conversation_to_text(self, conversation: Dict) -> str:
        """Convert conversation to training text."""
        text_parts = []
        
        for message in conversation["messages"]:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                text_parts.append(f"<|system|>\n{content}\n</s>")
            elif role == "user":
                text_parts.append(f"<|user|>\n{content}\n</s>")
            elif role == "assistant":
                text_parts.append(f"<|assistant|>\n{content}\n</s>")
        
        return "".join(text_parts)

@app.local_entrypoint()
def main(base_model: str = "codellama/CodeLlama-7b-hf"):
    """Main fine-tuning pipeline."""
    tuner = AdvancedOllamaFineTuner()
    
    # Prepare model
    tuner.prepare_model_and_tokenizer.remote(base_model)
    
    # Create dataset
    dataset_path = tuner.create_reasoning_dataset.remote()
    
    # Fine-tune
    model_path = tuner.fine_tune_model.remote(dataset_path)
    
    print(f"✅ Fine-tuning completed! Model available at: {model_path}")