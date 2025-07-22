#!/usr/bin/env python3
"""
Simple binary topology classification training with dimension fix
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config

def main():
    print("🚀 Starting Simple Binary Topology Classification Training")
    print("=" * 60)
    
    # Verify config
    print(f"Topology classes: {config.NUM_TOPOLOGY_CLASSES}")
    if config.NUM_TOPOLOGY_CLASSES != 2:
        print("❌ ERROR: Config not set for binary classification!")
        return
    
    print("✅ Configuration verified for binary classification")
    print("\n📋 Training Configuration:")
    print(f"  • Classes: Trivial (0) vs Topological (1)")
    print(f"  • Crystal node features: 3D (actual data)")
    print(f"  • K-space node features: 10D")
    print(f"  • Scalar features: 4763D")
    print(f"  • Batch size: {config.BATCH_SIZE}")
    print(f"  • Learning rate: {config.LEARNING_RATE}")
    print(f"  • Epochs: {config.NUM_EPOCHS}")
    print(f"  • Device: {config.DEVICE}")
    
    print("\n🔧 Components Enabled:")
    print("  • Legacy crystal encoder (3D → 256D)")
    print("  • K-space transformer encoder")
    print("  • Scalar feature encoder")
    print("  • Physics feature encoder")
    print("  • Spectral graph encoder")
    print("  • Persistent homology (ASPH)")
    print("  • Topological ML encoder")
    print("  • Dynamic fusion network")
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    # Import and run training
    try:
        from training.classifier_training import main_training_loop
        main_training_loop()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()