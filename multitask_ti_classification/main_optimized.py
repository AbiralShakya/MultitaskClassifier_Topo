#!/usr/bin/env python3
"""
Main entry point for optimized training
Use this instead of src/main.py to avoid dimension issues
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function that runs the smart optimized training"""
    print("🎯 Starting Optimized Training Pipeline")
    print("🚀 Using Smart Architecture with Existing Working Components")
    print("=" * 70)
    
    try:
        # Import and run smart training
        from run_smart_training import main as smart_main
        test_acc, test_f1 = smart_main()
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        
        if test_acc >= 0.92:
            print("🎉 SUCCESS: Achieved 92%+ test accuracy!")
        else:
            print(f"📈 Progress made. Need {(0.92 - test_acc)*100:.1f}% more accuracy.")
        
        return test_acc, test_f1
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()