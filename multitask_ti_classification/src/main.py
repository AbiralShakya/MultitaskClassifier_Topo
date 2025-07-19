print("[DEBUG] Script starting - main.py execution begins")
import sys
import os
print("[DEBUG] Imported sys and os")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("[DEBUG] Added path to sys.path")

print("[DEBUG] About to import crazy trainer...")
from training import crazy_training as trainer
print("[DEBUG] Successfully imported crazy trainer")

print("[DEBUG] About to import config...")
from helper import config
print("[DEBUG] Successfully imported config")

print("[DEBUG] About to import crazy fusion model...")
from src import crazy_fusion_model as model
print("[DEBUG] Successfully imported crazy fusion model")

import os
from pathlib import Path
print("[DEBUG] Imported os and Path")

def setup_environment():
    """Ensures necessary directories exist and checks for data."""
    print("Setting up environment...")
    
    # Create directory for saving models
    Path(config.MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Check if we're running with dummy data (for testing)
    use_dummy_data = False  # Set to False when you have real data
    
    if use_dummy_data:
        print("⚠️  Running with dummy data for testing purposes.")
        print("   Set use_dummy_data = False when you have real data files.")
    else:
        # Basic checks for data existence
        if not config.MASTER_INDEX_PATH.exists():
            print(f"ERROR: Master index not found at {config.MASTER_INDEX_PATH}.")
            print("Please ensure your data generation pipeline (IntegratedMaterialProcessor) has been run successfully.")
            exit(1)
        
        if not config.KSPACE_GRAPHS_DIR.exists():
            print(f"ERROR: K-space graphs directory not found at {config.KSPACE_GRAPHS_DIR}.")
            print("Please ensure KSpacePhysicsGraphBuilder has been run to pre-generate these graphs.")
            exit(1)
    
    print("Environment setup complete.")

if __name__ == "__main__":
    setup_environment()
    print("\nStarting crazy fusion model training for multi-modal material classification...")
    trainer.main_training_loop()
    print("\nTraining process finished.")