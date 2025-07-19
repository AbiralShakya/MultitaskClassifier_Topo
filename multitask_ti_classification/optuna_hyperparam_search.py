"""
Optuna Hyperparameter Search - Comprehensive hyperparameter optimization
for the crazy fusion model using Optuna.
"""

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.crazy_fusion_model import create_crazy_fusion_model
from training.crazy_training import create_crazy_trainer
from helper.config import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm


class OptunaObjective:
    """Objective function for Optuna hyperparameter optimization."""
    
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, 
                 device: str = 'cuda', n_epochs: int = 20):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_epochs = n_epochs
        self.best_trials = []
        
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function to minimize (negative validation accuracy)."""
        
        # Sample hyperparameters
        config = self._sample_hyperparameters(trial)
        
        try:
            # Create model and trainer
            model = create_crazy_fusion_model(config)
            trainer = create_crazy_trainer(model, config, self.device)
            
            # Train for limited epochs
            best_val_acc = 0.0
            
            for epoch in range(self.n_epochs):
                # Train one epoch
                train_loss, train_acc = trainer.train_epoch(self.train_loader)
                
                # Validate
                val_loss, val_acc, _ = trainer.validate(self.val_loader)
                
                # Update best validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                # Early stopping if validation accuracy is too low
                if epoch > 5 and best_val_acc < 70.0:
                    break
                
                # Update learning rate
                trainer.scheduler.step()
            
            # Store trial info
            trial.set_user_attr('best_val_acc', best_val_acc)
            trial.set_user_attr('config', config)
            
            # Return negative accuracy (Optuna minimizes)
            return -best_val_acc
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return 1000.0  # Large penalty for failed trials
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Sample hyperparameters for the trial."""
        
        config = {}
        
        # Model architecture
        config['HIDDEN_DIM'] = trial.suggest_categorical('hidden_dim', [128, 256, 512, 1024])
        config['FUSION_DIM'] = trial.suggest_categorical('fusion_dim', [256, 512, 1024, 2048])
        config['NUM_CLASSES'] = 2
        
        # Modality flags
        config['USE_CRYSTAL'] = trial.suggest_categorical('use_crystal', [True, False])
        config['USE_KSPACE'] = trial.suggest_categorical('use_kspace', [True, False])
        config['USE_SCALAR'] = trial.suggest_categorical('use_scalar', [True, False])
        config['USE_DECOMPOSITION'] = trial.suggest_categorical('use_decomposition', [True, False])
        config['USE_SPECTRAL'] = trial.suggest_categorical('use_spectral', [True, False])
        
        # Input dimensions
        config['CRYSTAL_INPUT_DIM'] = 92
        config['KSPACE_INPUT_DIM'] = 2
        config['SCALAR_INPUT_DIM'] = 200
        config['DECOMPOSITION_INPUT_DIM'] = 100
        config['K_EIGS'] = trial.suggest_categorical('k_eigs', [32, 64, 128, 256])
        
        # Architecture depths
        config['CRYSTAL_LAYERS'] = trial.suggest_int('crystal_layers', 2, 6)
        config['KSPACE_LAYERS'] = trial.suggest_int('kspace_layers', 2, 5)
        config['SCALAR_BLOCKS'] = trial.suggest_int('scalar_blocks', 2, 6)
        config['FUSION_BLOCKS'] = trial.suggest_int('fusion_blocks', 2, 5)
        config['FUSION_HEADS'] = trial.suggest_categorical('fusion_heads', [4, 8, 12, 16])
        
        # GNN types
        config['KSPACE_GNN_TYPE'] = trial.suggest_categorical('kspace_gnn_type', 
                                                             ['transformer', 'gat', 'gcn', 'sage'])
        
        # Training hyperparameters
        config['LEARNING_RATE'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        config['WEIGHT_DECAY'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        config['MIXUP_ALPHA'] = trial.suggest_float('mixup_alpha', 0.0, 0.5)
        config['CUTMIX_ALPHA'] = trial.suggest_float('cutmix_alpha', 0.0, 2.0)
        config['MASK_PROB'] = trial.suggest_float('mask_prob', 0.0, 0.3)
        
        # Loss function parameters
        config['FOCAL_ALPHA'] = trial.suggest_float('focal_alpha', 0.5, 2.0)
        config['FOCAL_GAMMA'] = trial.suggest_float('focal_gamma', 1.0, 4.0)
        
        # Learning rate scheduler
        config['SCHEDULER_T0'] = trial.suggest_int('scheduler_t0', 5, 20)
        config['SCHEDULER_T_MULT'] = trial.suggest_categorical('scheduler_t_mult', [1, 2])
        config['SCHEDULER_ETA_MIN'] = trial.suggest_float('scheduler_eta_min', 1e-7, 1e-5, log=True)
        
        # Other settings
        config['USE_WANDB'] = False
        config['SEED'] = trial.suggest_int('seed', 42, 1000)
        
        return config


class OptunaHyperparameterSearch:
    """Comprehensive hyperparameter search using Optuna."""
    
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, 
                 device: str = 'cuda', study_name: str = 'crazy_fusion_optimization'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.study_name = study_name
        self.study = None
        self.best_config = None
        self.best_accuracy = 0.0
        
    def run_optimization(self, n_trials: int = 100, n_epochs_per_trial: int = 20,
                        save_dir: str = 'optuna_results'):
        """Run hyperparameter optimization."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            storage=f'sqlite:///{save_dir}/optuna_study.db',
            load_if_exists=True
        )
        
        # Create objective
        objective = OptunaObjective(
            self.train_loader, 
            self.val_loader, 
            self.device, 
            n_epochs_per_trial
        )
        
        print(f"Starting hyperparameter optimization with {n_trials} trials...")
        print(f"Each trial will train for {n_epochs_per_trial} epochs")
        
        # Run optimization
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best results
        self.best_config = self.study.best_trial.user_attrs['config']
        self.best_accuracy = -self.study.best_value
        
        print(f"\nOptimization completed!")
        print(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        print(f"Best config: {self.best_config}")
        
        # Save results
        self.save_results(save_dir)
        
        return self.best_config, self.best_accuracy
    
    def save_results(self, save_dir: str):
        """Save optimization results."""
        
        # Save best config
        with open(os.path.join(save_dir, 'best_config.json'), 'w') as f:
            json.dump(self.best_config, f, indent=2)
        
        # Save study
        joblib.dump(self.study, os.path.join(save_dir, 'study.pkl'))
        
        # Create results summary
        results = {
            'best_accuracy': self.best_accuracy,
            'best_config': self.best_config,
            'n_trials': len(self.study.trials),
            'study_name': self.study_name
        }
        
        with open(os.path.join(save_dir, 'results_summary.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot optimization history
        self.plot_optimization_history(save_dir)
        
        # Plot parameter importance
        self.plot_parameter_importance(save_dir)
        
        # Plot parameter relationships
        self.plot_parameter_relationships(save_dir)
    
    def plot_optimization_history(self, save_dir: str):
        """Plot optimization history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
        ax1.set_title('Optimization History')
        
        # Parameter importance
        optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
        ax2.set_title('Parameter Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_parameter_importance(self, save_dir: str):
        """Plot parameter importance."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax)
            ax.set_title('Hyperparameter Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'parameter_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not plot parameter importance: {e}")
    
    def plot_parameter_relationships(self, save_dir: str):
        """Plot parameter relationships."""
        try:
            # Get top parameters
            importances = optuna.importance.get_param_importances(self.study)
            top_params = list(importances.keys())[:6]  # Top 6 parameters
            
            if len(top_params) >= 2:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                
                for i, param in enumerate(top_params[:6]):
                    if i < len(axes):
                        optuna.visualization.matplotlib.plot_param_importances(
                            self.study, target=lambda t: t.params.get(param, 0), ax=axes[i]
                        )
                        axes[i].set_title(f'{param} vs Objective')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'parameter_relationships.png'), dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Could not plot parameter relationships: {e}")
    
    def get_top_configs(self, n: int = 5) -> List[Tuple[Dict, float]]:
        """Get top n configurations."""
        top_configs = []
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                config = trial.user_attrs['config']
                accuracy = trial.user_attrs['best_val_acc']
                top_configs.append((config, accuracy))
        
        # Sort by accuracy
        top_configs.sort(key=lambda x: x[1], reverse=True)
        
        return top_configs[:n]
    
    def analyze_results(self, save_dir: str):
        """Analyze optimization results."""
        
        print("\n" + "="*50)
        print("HYPERPARAMETER OPTIMIZATION ANALYSIS")
        print("="*50)
        
        # Best configuration
        print(f"\nBest Configuration:")
        print(f"  Validation Accuracy: {self.best_accuracy:.2f}%")
        for key, value in self.best_config.items():
            print(f"  {key}: {value}")
        
        # Top configurations
        top_configs = self.get_top_configs(5)
        print(f"\nTop 5 Configurations:")
        for i, (config, accuracy) in enumerate(top_configs):
            print(f"\n{i+1}. Accuracy: {accuracy:.2f}%")
            print(f"   Key differences from best:")
            for key, value in config.items():
                if key in self.best_config and value != self.best_config[key]:
                    print(f"     {key}: {value} (vs {self.best_config[key]})")
        
        # Parameter importance
        try:
            importances = optuna.importance.get_param_importances(self.study)
            print(f"\nParameter Importance:")
            for param, importance in importances.items():
                print(f"  {param}: {importance:.4f}")
        except Exception as e:
            print(f"Could not compute parameter importance: {e}")
        
        # Save analysis
        analysis = {
            'best_config': self.best_config,
            'best_accuracy': self.best_accuracy,
            'top_configs': [(config, acc) for config, acc in top_configs],
            'n_trials': len(self.study.trials)
        }
        
        with open(os.path.join(save_dir, 'analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\nAnalysis saved to {save_dir}/analysis.json")


def run_hyperparameter_search(train_loader: DataLoader, val_loader: DataLoader,
                             n_trials: int = 100, n_epochs_per_trial: int = 20,
                             save_dir: str = 'optuna_results') -> Tuple[Dict, float]:
    """Run complete hyperparameter search pipeline."""
    
    # Create optimizer
    optimizer = OptunaHyperparameterSearch(
        train_loader, val_loader, study_name='crazy_fusion_optimization'
    )
    
    # Run optimization
    best_config, best_accuracy = optimizer.run_optimization(
        n_trials=n_trials, 
        n_epochs_per_trial=n_epochs_per_trial,
        save_dir=save_dir
    )
    
    # Analyze results
    optimizer.analyze_results(save_dir)
    
    return best_config, best_accuracy


if __name__ == "__main__":
    # Test hyperparameter sampling
    print("Testing hyperparameter sampling...")
    
    # Create a dummy trial
    study = optuna.create_study(direction='minimize')
    trial = study.ask()
    
    # Create objective
    objective = OptunaObjective(None, None)  # Dummy loaders
    
    # Sample hyperparameters
    config = objective._sample_hyperparameters(trial)
    
    print("Sampled configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nConfiguration has {sum(p.numel() for p in create_crazy_fusion_model(config).parameters()):,} parameters") 