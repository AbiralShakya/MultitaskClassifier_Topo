import training.ssl_trainer as train
import helper.config as config
import os
from pathlib import Path

def setup_environment():
    """Ensures necessary directories exist and checks for data."""
    print("Setting up environment...")
    
    # Create directory for saving models
    Path(config.MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
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
    print("\nStarting multi-modal material classification training...")
    train.main_training_loop()
    print("\nTraining process finished.")