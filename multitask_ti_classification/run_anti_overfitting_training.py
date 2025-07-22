#!/usr/bin/env python3
"""
Anti-overfitting training script with enhanced regularization
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper.config as config

def main():
    print("🛡️ Starting Anti-Overfitting Binary Topology Classification Training")
    print("=" * 70)
    
    # Verify anti-overfitting config
    print("📋 Anti-Overfitting Configuration:")
    print(f"  • Learning Rate: {config.LEARNING_RATE} (reduced)")
    print(f"  • Weight Decay: {config.WEIGHT_DECAY} (increased L2 reg)")
    print(f"  • Batch Size: {config.BATCH_SIZE} (smaller for better generalization)")
    print(f"  • Dropout Rate: {config.DROPOUT_RATE} (increased)")
    print(f"  • Patience: {config.PATIENCE} (reduced for early stopping)")
    print(f"  • Max Epochs: {config.NUM_EPOCHS} (reduced)")
    
    print("\n🛡️ Anti-Overfitting Techniques Enabled:")
    print("  • Stronger L2 regularization (10x weight decay)")
    print("  • Higher dropout rates (0.6 in fusion network)")
    print("  • Feature noise augmentation during training")
    print("  • Smaller batch size for better generalization")
    print("  • Earlier stopping (patience=8)")
    print("  • Deeper fusion network with more dropout layers")
    print("  • Aggressive learning rate scheduling")
    
    print("\n🎯 Expected Behavior:")
    print("  • Training and validation loss should decrease together")
    print("  • Gap between train/val accuracy should be <3%")
    print("  • Model should stop training when validation stops improving")
    print("  • Better generalization to test set")
    
    print("\n" + "=" * 70)
    print("Starting anti-overfitting training...")
    print("=" * 70)
    
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