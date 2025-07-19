"""
Hyperparameter Optimization for Crazy Fusion Model
=================================================

This module provides comprehensive hyperparameter optimization using Optuna
for the multimodal transformer fusion model for topological material classification.
"""

import optuna
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, Any, Optional, Tuple
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training.crazy_training import CrazyTrainer, create_crazy_fusion_model
    from data.real_data_loaders import create_data_loaders, get_class_weights
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Optimization dependencies not available: {e}")
    OPTIMIZATION_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Wandb not available. Logging will be disabled.")


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimizer for the Crazy Fusion Model.
    """
    
    def __init__(self, 
                 data_dir: Path,
                 study_name: str = "crazy_fusion_optimization",
                 storage: Optional[str] = None,
                 n_trials: int = 100,
                 timeout: Optional[int] = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            data_dir: Directory containing processed data
            study_name: Name for the Optuna study
            storage: Optuna storage URL (e.g., "sqlite:///optuna.db")
            n_trials: Number of optimization trials
            timeout: Timeout in seconds for the entire optimization
        """
        if not OPTIMIZATION_AVAILABLE:
            raise ImportError("Required dependencies not available for optimization")
        
        self.data_dir = Path(data_dir)
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        
        # Create study
        if storage:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",  # Maximize validation accuracy
                load_if_exists=True
            )
        else:
            self.study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                load_if_exists=True
            )
        
        # Best trial tracking
        self.best_trial = None
        self.best_config = None
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        config = {}
        
        # Model architecture hyperparameters
        config['HIDDEN_DIM'] = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        config['FUSION_DIM'] = trial.suggest_categorical('fusion_dim', [256, 512, 1024])
        config['NUM_CLASSES'] = 3  # Fixed for our task
        
        # Modality usage
        config['USE_CRYSTAL'] = trial.suggest_categorical('use_crystal', [True, False])
        config['USE_KSPACE'] = trial.suggest_categorical('use_kspace', [True, False])
        config['USE_SCALAR'] = trial.suggest_categorical('use_scalar', [True, False])
        config['USE_DECOMPOSITION'] = trial.suggest_categorical('use_decomposition', [True, False])
        config['USE_SPECTRAL'] = trial.suggest_categorical('use_spectral', [True, False])
        
        # Input dimensions (fixed based on data)
        config['CRYSTAL_INPUT_DIM'] = 92
        config['KSPACE_INPUT_DIM'] = 2
        config['SCALAR_INPUT_DIM'] = 200
        config['DECOMPOSITION_INPUT_DIM'] = 100
        config['K_EIGS'] = trial.suggest_categorical('k_eigs', [32, 64, 128])
        
        # Layer configurations
        config['CRYSTAL_LAYERS'] = trial.suggest_int('crystal_layers', 2, 6)
        config['KSPACE_LAYERS'] = trial.suggest_int('kspace_layers', 2, 5)
        config['SCALAR_BLOCKS'] = trial.suggest_int('scalar_blocks', 2, 5)
        config['FUSION_BLOCKS'] = trial.suggest_int('fusion_blocks', 2, 5)
        config['FUSION_HEADS'] = trial.suggest_categorical('fusion_heads', [4, 8, 16])
        
        # GNN type for k-space
        config['KSPACE_GNN_TYPE'] = trial.suggest_categorical('kspace_gnn_type', 
                                                             ['transformer', 'gat', 'gcn', 'sage'])
        
        # Training hyperparameters
        config['LEARNING_RATE'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        config['WEIGHT_DECAY'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        config['BATCH_SIZE'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Loss function parameters
        config['FOCAL_ALPHA'] = trial.suggest_float('focal_alpha', 0.5, 2.0)
        config['FOCAL_GAMMA'] = trial.suggest_float('focal_gamma', 1.0, 5.0)
        
        # Scheduler parameters
        config['SCHEDULER_T0'] = trial.suggest_int('scheduler_t0', 5, 30)
        config['SCHEDULER_T_MULT'] = trial.suggest_categorical('scheduler_t_mult', [1, 2])
        config['SCHEDULER_ETA_MIN'] = trial.suggest_float('scheduler_eta_min', 1e-7, 1e-5, log=True)
        
        # Data augmentation parameters
        config['MIXUP_ALPHA'] = trial.suggest_float('mixup_alpha', 0.1, 0.5)
        config['CUTMIX_PROB'] = trial.suggest_float('cutmix_prob', 0.3, 0.7)
        config['FEATURE_MASK_PROB'] = trial.suggest_float('feature_mask_prob', 0.05, 0.2)
        config['EDGE_DROPOUT'] = trial.suggest_float('edge_dropout', 0.05, 0.2)
        config['NODE_FEATURE_NOISE'] = trial.suggest_float('node_feature_noise', 0.01, 0.1)
        
        # Data loading parameters
        config['NUM_WORKERS'] = trial.suggest_int('num_workers', 2, 8)
        config['MAX_CRYSTAL_NODES'] = trial.suggest_categorical('max_crystal_nodes', [500, 1000, 2000])
        config['MAX_KSPACE_NODES'] = trial.suggest_categorical('max_kspace_nodes', [250, 500, 1000])
        
        # Other settings
        config['USE_WANDB'] = WANDB_AVAILABLE and trial.suggest_categorical('use_wandb', [True, False])
        config['SEED'] = trial.suggest_int('seed', 42, 999)
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy (to be maximized)
        """
        # Suggest hyperparameters
        config = self.suggest_hyperparameters(trial)
        
        # Set random seed
        torch.manual_seed(config['SEED'])
        np.random.seed(config['SEED'])
        
        try:
            # Create model
            model = create_crazy_fusion_model(config)
            
            # Create trainer
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            trainer = CrazyTrainer(model, config, device=device)
            
            # Create data loaders
            modalities = []
            if config['USE_CRYSTAL']:
                modalities.append('crystal')
            if config['USE_KSPACE']:
                modalities.append('kspace')
            if config['USE_SCALAR']:
                modalities.append('scalar')
            if config['USE_DECOMPOSITION']:
                modalities.append('decomposition')
            if config['USE_SPECTRAL']:
                modalities.append('spectral')
            
            if not modalities:
                # At least one modality must be enabled
                return 0.0
            
            loaders = create_data_loaders(
                data_dir=self.data_dir,
                batch_size=config['BATCH_SIZE'],
                modalities=modalities,
                num_workers=config['NUM_WORKERS'],
                augment=True,
                mixup_alpha=config['MIXUP_ALPHA'],
                cutmix_prob=config['CUTMIX_PROB'],
                feature_mask_prob=config['FEATURE_MASK_PROB'],
                edge_dropout=config['EDGE_DROPOUT'],
                node_feature_noise=config['NODE_FEATURE_NOISE']
            )
            
            train_loader = loaders['train']
            val_loader = loaders['val']
            
            # Train for a limited number of epochs for optimization
            num_epochs = min(20, len(train_loader) // 10)  # Adaptive epochs
            
            best_val_acc = 0.0
            
            for epoch in range(num_epochs):
                # Train one epoch
                train_loss, train_acc = trainer.train_epoch(train_loader)
                
                # Validate
                val_loss, val_acc, _ = trainer.validate(val_loader)
                
                # Report intermediate value
                trial.report(val_acc, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                best_val_acc = max(best_val_acc, val_acc)
            
            return best_val_acc
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return 0.0
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        Run the hyperparameter optimization.
        
        Returns:
            Tuple of (best_config, best_value)
        """
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        print(f"Study name: {self.study_name}")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best results
        self.best_trial = self.study.best_trial
        self.best_config = self.best_trial.params
        best_value = self.best_trial.value
        
        print(f"\nOptimization completed!")
        print(f"Best validation accuracy: {best_value:.4f}")
        print(f"Best trial number: {self.best_trial.number}")
        
        return self.best_config, best_value
    
    def save_results(self, output_dir: Path):
        """
        Save optimization results.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best configuration
        config_path = output_dir / "best_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.best_config, f, indent=2)
        
        # Save study
        study_path = output_dir / "study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        # Save optimization history
        history_path = output_dir / "optimization_history.json"
        history = {
            'trials': [],
            'best_trial': self.best_trial.number if self.best_trial else None,
            'best_value': self.best_trial.value if self.best_trial else None
        }
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history['trials'].append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'duration': trial.duration.total_seconds()
                })
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Results saved to {output_dir}")
    
    def plot_optimization_history(self, output_dir: Path):
        """
        Plot optimization history.
        
        Args:
            output_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot optimization history
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title('Optimization History')
            
            # Parameter importance
            try:
                optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
                ax2.set_title('Parameter Importance')
            except:
                ax2.text(0.5, 0.5, 'Parameter importance not available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Parameter Importance')
            
            # Parameter relationships
            try:
                optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=ax3)
                ax3.set_title('Parameter Relationships')
            except:
                ax3.text(0.5, 0.5, 'Parameter relationships not available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Parameter Relationships')
            
            # Parameter contour
            try:
                # Get two most important parameters
                importances = optuna.importance.get_param_importances(self.study)
                if len(importances) >= 2:
                    param1, param2 = list(importances.keys())[:2]
                    optuna.visualization.matplotlib.plot_contour(self.study, params=[param1, param2], ax=ax4)
                    ax4.set_title(f'Contour: {param1} vs {param2}')
                else:
                    ax4.text(0.5, 0.5, 'Contour plot not available', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Parameter Contour')
            except:
                ax4.text(0.5, 0.5, 'Contour plot not available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Parameter Contour')
            
            plt.tight_layout()
            plt.savefig(output_dir / "optimization_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Optimization plots saved to {output_dir}")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Error creating plots: {e}")


def run_hyperparameter_optimization(data_dir: Path,
                                  study_name: str = "crazy_fusion_optimization",
                                  n_trials: int = 100,
                                  timeout: Optional[int] = None,
                                  output_dir: Path = Path("optimization_results")):
    """
    Run complete hyperparameter optimization pipeline.
    
    Args:
        data_dir: Directory containing processed data
        study_name: Name for the Optuna study
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        output_dir: Directory to save results
    """
    if not OPTIMIZATION_AVAILABLE:
        print("❌ Optimization dependencies not available")
        return None, 0.0
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        data_dir=data_dir,
        study_name=study_name,
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Run optimization
    best_config, best_value = optimizer.optimize()
    
    # Save results
    optimizer.save_results(output_dir)
    optimizer.plot_optimization_history(output_dir)
    
    return best_config, best_value


if __name__ == "__main__":
    # Example usage
    data_dir = Path("data/processed")
    output_dir = Path("optimization_results")
    
    if data_dir.exists():
        print("🚀 Starting hyperparameter optimization...")
        best_config, best_value = run_hyperparameter_optimization(
            data_dir=data_dir,
            study_name="crazy_fusion_test",
            n_trials=20,  # Small number for testing
            timeout=3600,  # 1 hour timeout
            output_dir=output_dir
        )
        
        if best_config:
            print(f"\n🎉 Optimization completed!")
            print(f"Best validation accuracy: {best_value:.4f}")
            print("Best configuration:")
            for key, value in best_config.items():
                print(f"  {key}: {value}")
    else:
        print(f"❌ Data directory {data_dir} not found")
        print("Please run data preprocessing first.") 