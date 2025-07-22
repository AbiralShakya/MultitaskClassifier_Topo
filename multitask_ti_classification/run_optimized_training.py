#!/usr/bin/env python3
"""
Run the optimized training pipeline integrated into main workflow
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.classifier_training import main_training_loop

if __name__ == "__main__":
    print("🎯 Target: 92%+ Test Accuracy Without Overfitting")
    print("🏗️  Architecture: Optimized Multi-Modal Classifier")
    print("🔧 Features: Focal Loss, Self-Attention, Residual Connections")
    print("📊 Training: Cosine Warmup, Early Stopping, Stratified Split")
    print("="*70)
    
    try:
        result = main_training_loop()
        
        if isinstance(result, tuple):
            test_acc, test_f1 = result
            if test_acc >= 0.92:
                print("\n🎉 SUCCESS! Target achieved!")
            else:
                print(f"\n📈 Progress made. Need {(0.92 - test_acc)*100:.1f}% more accuracy.")
        else:
            print("\n✅ Training completed successfully!")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()