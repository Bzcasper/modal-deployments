"""
🔍 System Requirements Checker
This script checks if your system has everything needed for AI model fine-tuning.
"""

import sys
import subprocess
import pkg_resources
import platform
import importlib
from pathlib import Path

def print_status(message, status="info"):
    """Print colored status messages."""
    colors = {
        "info": "\033[94m[INFO]\033[0m",
        "success": "\033[92m[SUCCESS]\033[0m", 
        "warning": "\033[93m[WARNING]\033[0m",
        "error": "\033[91m[ERROR]\033[0m"
    }
    print(f"{colors.get(status, '')} {message}")

def check_python_version():
    """Check if Python version is compatible."""
    print_status("Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} ✓", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Need 3.8+", "error")
        return False

def check_required_packages():
    """Check if all required packages are installed."""
    print_status("Checking required packages...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'modal', 
        'peft', 'accelerate', 'wandb', 'pyyaml', 'loguru'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_status(f"  {package} ✓", "success")
        except ImportError:
            print_status(f"  {package} ✗", "error")
            missing_packages.append(package)
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "error")
        print_status("Run: pip install -r requirements.txt", "info")
        return False
    
    return True

def check_system_resources():
    """Check system resources and capabilities."""
    print_status("Checking system resources...")
    
    # Check available RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print_status(f"Available RAM: {ram_gb:.1f} GB", "success" if ram_gb >= 8 else "warning")
        
        if ram_gb < 8:
            print_status("Recommendation: 8GB+ RAM for better performance", "warning")
    except ImportError:
        print_status("Cannot check RAM (psutil not installed)", "warning")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_status(f"CUDA GPU available: {gpu_name} ({gpu_count} GPU(s))", "success")
        else:
            print_status("No CUDA GPU detected - will use CPU (slower)", "warning")
    except ImportError:
        print_status("Cannot check GPU (torch not installed)", "warning")

def check_directory_structure():
    """Check if project directories exist."""
    print_status("Checking project structure...")
    
    required_dirs = [
        'src', 'config', 'examples', 'tests', 'docs', 
        'data/raw', 'data/processed', 'models/checkpoints'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_status(f"  {dir_path}/ ✓", "success")
        else:
            print_status(f"  {dir_path}/ ✗", "error")
            all_good = False
    
    return all_good

def check_environment_files():
    """Check if configuration files exist."""
    print_status("Checking configuration files...")
    
    config_files = [
        'requirements.txt',
        'config/training_config.yaml',
        'config/model_config.yaml',
        '.env.template'
    ]
    
    all_good = True
    for file_path in config_files:
        path = Path(file_path)
        if path.exists():
            print_status(f"  {file_path} ✓", "success")
        else:
            print_status(f"  {file_path} ✗", "error")
            all_good = False
    
    return all_good

def main():
    """Run all system checks."""
    print("🔍 AI Model Fine-tuning System Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("System Resources", check_system_resources),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_environment_files)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        result = check_func()
        if result is False:
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print_status("🎉 All checks passed! Your system is ready!", "success")
        return 0
    else:
        print_status("❌ Some checks failed. Please fix the issues above.", "error")
        return 1

if __name__ == "__main__":
    sys.exit(main())