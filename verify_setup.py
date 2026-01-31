"""
Verification script to check that the fraud detection system is properly set up.
Run this after installing dependencies with: pip install -r requirements.txt
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check that all required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'lightgbm', 'xgboost', 'hypothesis'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def check_project_structure():
    """Check that project structure is correct."""
    required_dirs = ['src', 'tests', 'data', 'models', 'logs']
    required_files = ['requirements.txt', 'setup.py', 'README.md', 'src/config.py']
    
    missing_items = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_items.append(f"directory: {dir_name}")
    
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_items.append(f"file: {file_name}")
    
    if missing_items:
        print(f"❌ Missing items: {', '.join(missing_items)}")
        return False
    else:
        print("✅ Project structure is correct")
        return True

def check_configuration():
    """Check that configuration module works."""
    try:
        sys.path.insert(0, 'src')
        from config import setup_logging, MODEL_CONFIG
        setup_logging()
        print(f"✅ Configuration loaded (random_seed: {MODEL_CONFIG['random_seed']})")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Verifying fraud detection system setup...\n")
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Configuration", check_configuration),
        ("Dependencies", check_dependencies),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"Checking {name}...")
        if not check_func():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 Setup verification completed successfully!")
        print("You can now proceed with implementing the fraud detection system.")
    else:
        print("❌ Setup verification failed. Please fix the issues above.")
        sys.exit(1)