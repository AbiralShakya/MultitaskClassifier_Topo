#!/usr/bin/env python3
"""
Run Hyperparameter Optimization for Crazy Fusion Model
=====================================================

This script runs comprehensive hyperparameter optimization for the multimodal
transformer fusion model for topological material classification.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimization.hyperparameter_optimization import run_hyperparameter_optimization
from helper import config

def main():
    """Run hyperparameter optimization."""
    print("🚀 Crazy Fusion Model Hyperparameter Optimization")
    print("=" * 50)
    
    # Check if data is available
    data_dir = config.DATA_DIR / "processed"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure you have processed data available.")
        print("Expected structure:")
        print("  data/processed/")
        print("  ├── train_index.json")
        print("  ├── val_index.json")
        print("  ├── test_index.json")
        print("  ├── labels.json")
        print("  ├── crystal_graphs/")
        print("  ├── kspace_graphs/")
        print("  ├── scalar_features/")
        print("  ├── decomposition_features/")
        print("  └── spectral_features/")
        return
    
    # Create output directory
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run optimization
    print(f"📊 Starting optimization with data from: {data_dir}")
    print(f"💾 Results will be saved to: {output_dir}")
    
    best_config, best_value = run_hyperparameter_optimization(
        data_dir=data_dir,
        study_name="crazy_fusion_optimization",
        n_trials=50,  # Adjust based on your computational budget
        timeout=7200,  # 2 hours timeout
        output_dir=output_dir
    )
    
    if best_config:
        print("\n🎉 Optimization completed successfully!")
        print(f"Best validation accuracy: {best_value:.4f}")
        print("\nBest configuration:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        
        # Save the best config for easy access
        config_path = output_dir / "best_config_for_training.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\n💾 Best configuration saved to: {config_path}")
        
        print("\n📝 To use this configuration for training:")
        print("1. Copy the best_config_for_training.json file")
        print("2. Use it in your training script")
        print("3. Or run: python src/main.py --config optimization_results/best_config_for_training.json")
        
    else:
        print("❌ Optimization failed or no valid trials completed")


if __name__ == "__main__":
    main() 