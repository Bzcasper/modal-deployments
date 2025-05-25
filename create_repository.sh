#!/bin/bash

# 🚀 AI Model Fine-tuning Repository Creator
# This script creates the complete repository structure and all files

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
REPO_NAME="ai-model-finetuning"
REPO_DESCRIPTION="🤖 Complete AI Model Fine-tuning System for Ollama with Modal Labs - High School Friendly Guide"

print_status "Creating AI Model Fine-tuning Repository..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Check if GitHub CLI is installed (optional but recommended)
if command -v gh &> /dev/null; then
    HAS_GH_CLI=true
    print_success "GitHub CLI detected - will help with repository creation"
else
    HAS_GH_CLI=false
    print_warning "GitHub CLI not found - you'll need to create the repository manually"
fi

# Create project directory
if [ -d "$REPO_NAME" ]; then
    print_warning "Directory $REPO_NAME already exists"
    read -p "Do you want to overwrite it? (y/N): " overwrite
    if [[ $overwrite != "y" && $overwrite != "Y" ]]; then
        print_error "Aborted"
        exit 1
    fi
    rm -rf "$REPO_NAME"
fi

mkdir -p "$REPO_NAME"
cd "$REPO_NAME"

print_success "Created project directory: $REPO_NAME"

# Create directory structure
print_status "Creating directory structure..."
mkdir -p {setup,src,config,examples,tests,docs,scripts,.github/workflows,data/{raw,processed},models/{checkpoints,final},logs}

# Create main README.md
print_status "Creating README.md..."
cat > README.md << 'EOL'
# 🚀 AI Model Fine-Tuning System
## A Complete Guide for High School Students

### What is this project?
This project helps you create your own AI coding assistant by fine-tuning (customizing) existing AI models to be better at helping with programming tasks. Think of it like teaching an AI to be a really good coding tutor!

### What you'll learn:
- How AI models work and how to improve them
- Cloud computing with Modal Labs
- GitHub automation
- Python programming
- AI/ML concepts made simple

---

## 📋 Prerequisites (What you need before starting)

### Required Knowledge:
- Basic Python programming (variables, functions, loops)
- Using command line/terminal
- Basic understanding of Git and GitHub

### Required Accounts:
1. **GitHub Account** (free) - for storing your code
2. **Modal Labs Account** (free tier available) - for running AI training
3. **Hugging Face Account** (free) - for accessing AI models
4. **Weights & Biases Account** (free) - for monitoring training

### Required Software:
- Python 3.8 or newer
- Git
- Visual Studio Code (recommended)
- Terminal/Command Prompt

---

## 🏗️ Project Structure
