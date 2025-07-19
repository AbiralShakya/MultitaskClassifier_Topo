#!/usr/bin/env python3
"""
Simple training script without hardcoded paths.
This script runs the training using the current directory structure.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    print("🚀 Starting training with local paths...")
    
    # Import the training module
    try:
        from training import crazy_training as trainer
        print("✅ Successfully imported training module")
    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
        return
    
    # Run the training
    try:
        trainer.main_training_loop()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 