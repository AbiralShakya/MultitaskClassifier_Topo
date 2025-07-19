"""
Crazy ML Pipeline - Master script that orchestrates the entire state-of-the-art
ML pipeline for topological insulator classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import argparse
from typing import Dict, List, Tuple
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.crazy_fusion_model import create_crazy_fusion_model
from training.crazy_training import create_crazy_trainer
from ensemble_training import train_ensemble_pipeline
# Hyperparameter optimization removed - using default configurations
from baseline_models import run_baseline_pipeline
from self_supervised_pretraining import run_pretraining_pipeline
from automated_analysis import run_automated_analysis
from helper.config import *


class CrazyMLPipeline:
    """Master pipeline orchestrating all ML components."""
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.results = {}
        
    def run_complete_pipeline(self, train_loader: DataLoader, val_loader: DataLoader, 
                            test_loader: DataLoader, save_dir: str = 'crazy_pipeline_results'):
        """Run the complete crazy ML pipeline."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("🚀 STARTING CRAZY ML PIPELINE 🚀")
        print("=" * 60)
        
        # Step 1: Self-supervised pretraining (optional)
        if self.config.get('RUN_PRETRAINING', False):
            print("\n📚 Step 1: Self-supervised pretraining...")
            pretrained_models = run_pretraining_pipeline(
                train_loader, val_loader, self.config, 
                os.path.join(save_dir, 'pretrained_models')
            )
            self.results['pretraining'] = pretrained_models
        
        # Step 2: Hyperparameter optimization (removed)
        print("\n⏭️  Hyperparameter optimization removed - using default configurations")
        best_config = self.config
        
        # Step 3: Train baseline models
        if self.config.get('RUN_BASELINES', True):
            print("\n📊 Step 3: Training baseline models...")
            baseline_results = run_baseline_pipeline(
                train_loader, val_loader, test_loader,
                save_dir=os.path.join(save_dir, 'baseline_models')
            )
            self.results['baselines'] = baseline_results
        
        # Step 4: Train ensemble models
        if self.config.get('RUN_ENSEMBLE', True):
            print("\n🎯 Step 4: Training ensemble models...")
            ensemble, ensemble_results = train_ensemble_pipeline(
                train_loader, val_loader, test_loader,
                [label for _, label in test_loader.dataset],
                num_epochs=self.config.get('ENSEMBLE_EPOCHS', 50),
                save_dir=os.path.join(save_dir, 'ensemble_models')
            )
            self.results['ensemble'] = ensemble_results
        
        # Step 5: Train final model with best config
        print("\n🏆 Step 5: Training final model...")
        final_model = create_crazy_fusion_model(self.config)
        final_trainer = create_crazy_trainer(final_model, self.config, self.device)
        
        final_trainer.train(
            train_loader, val_loader, 
            num_epochs=self.config.get('FINAL_EPOCHS', 100),
            save_path=os.path.join(save_dir, 'final_model.pth')
        )
        
        # Step 6: Automated analysis
        if self.config.get('RUN_ANALYSIS', True):
            print("\n📈 Step 6: Automated analysis...")
            analysis_results = run_automated_analysis(
                final_model, train_loader, val_loader, test_loader,
                save_dir=os.path.join(save_dir, 'analysis')
            )
            self.results['analysis'] = analysis_results
        
        # Step 7: Generate comprehensive report
        print("\n📋 Step 7: Generating comprehensive report...")
        self._generate_comprehensive_report(save_dir)
        
        print("\n🎉 CRAZY ML PIPELINE COMPLETED! 🎉")
        print(f"Results saved to: {save_dir}")
        
        return self.results
    
    def _generate_comprehensive_report(self, save_dir: str):
        """Generate comprehensive pipeline report."""
        
        report = {
            'pipeline_config': self.config,
            'results': self.results,
            'summary': self._create_summary()
        }
        
        # Save report
        with open(os.path.join(save_dir, 'comprehensive_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary text file
        with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
            f.write(self._create_summary_text())
    
    def _create_summary(self) -> Dict:
        """Create summary of all results."""
        
        summary = {}
        
        # Hyperparameter search results (removed)
        # summary['best_hyperparam_accuracy'] = "Hyperparameter optimization removed"
        
        # Baseline results
        if 'baselines' in self.results:
            baseline_accuracies = {}
            for model_name, results in self.results['baselines'].items():
                baseline_accuracies[model_name] = results['accuracy']
            summary['baseline_accuracies'] = baseline_accuracies
            summary['best_baseline'] = max(baseline_accuracies.items(), key=lambda x: x[1])
        
        # Ensemble results
        if 'ensemble' in self.results:
            ensemble_accuracies = {}
            for method, results in self.results['ensemble'].items():
                ensemble_accuracies[method] = results['accuracy']
            summary['ensemble_accuracies'] = ensemble_accuracies
            summary['best_ensemble'] = max(ensemble_accuracies.items(), key=lambda x: x[1])
        
        return summary
    
    def _create_summary_text(self) -> str:
        """Create human-readable summary text."""
        
        text = "CRAZY ML PIPELINE SUMMARY\n"
        text += "=" * 50 + "\n\n"
        
        # Configuration summary
        text += "CONFIGURATION:\n"
        text += f"  Hidden Dimension: {self.config.get('HIDDEN_DIM', 'N/A')}\n"
        text += f"  Fusion Dimension: {self.config.get('FUSION_DIM', 'N/A')}\n"
        text += f"  Learning Rate: {self.config.get('LEARNING_RATE', 'N/A')}\n"
        text += f"  Modalities: Crystal={self.config.get('USE_CRYSTAL', False)}, "
        text += f"K-space={self.config.get('USE_KSPACE', False)}, "
        text += f"Scalar={self.config.get('USE_SCALAR', False)}, "
        text += f"Decomposition={self.config.get('USE_DECOMPOSITION', False)}, "
        text += f"Spectral={self.config.get('USE_SPECTRAL', False)}\n\n"
        
        # Results summary
        text += "RESULTS:\n"
        
        # Hyperparameter search results removed
        # text += f"  Best Hyperparameter Accuracy: {self.results['hyperparam_search']['best_accuracy']:.4f}\n"
        
        if 'baselines' in self.results:
            text += "  Baseline Models:\n"
            for model_name, results in self.results['baselines'].items():
                text += f"    {model_name}: {results['accuracy']:.4f}\n"
        
        if 'ensemble' in self.results:
            text += "  Ensemble Models:\n"
            for method, results in self.results['ensemble'].items():
                text += f"    {method}: {results['accuracy']:.4f}\n"
        
        text += "\nCONCLUSIONS:\n"
        
        # Find best overall result
        best_accuracy = 0.0
        best_method = "Unknown"
        
        if 'ensemble' in self.results:
            for method, results in self.results['ensemble'].items():
                if results['accuracy'] > best_accuracy:
                    best_accuracy = results['accuracy']
                    best_method = f"Ensemble ({method})"
        
        if 'baselines' in self.results:
            for model_name, results in self.results['baselines'].items():
                if results['accuracy'] > best_accuracy:
                    best_accuracy = results['accuracy']
                    best_method = f"Baseline ({model_name})"
        
        text += f"  Best Overall Accuracy: {best_accuracy:.4f} ({best_method})\n"
        
        return text


def create_crazy_config() -> Dict:
    """Create a comprehensive configuration for the crazy pipeline."""
    
    config = {
        # Model architecture
        'HIDDEN_DIM': 512,
        'FUSION_DIM': 1024,
        'NUM_CLASSES': 2,
        'USE_CRYSTAL': True,
        'USE_KSPACE': True,
        'USE_SCALAR': True,
        'USE_DECOMPOSITION': True,
        'USE_SPECTRAL': True,
        'CRYSTAL_INPUT_DIM': 92,
        'KSPACE_INPUT_DIM': 2,
        'SCALAR_INPUT_DIM': 4763,
        'DECOMPOSITION_INPUT_DIM': 100,
        'K_EIGS': 128,
        'CRYSTAL_LAYERS': 5,
        'KSPACE_LAYERS': 4,
        'SCALAR_BLOCKS': 4,
        'FUSION_BLOCKS': 4,
        'FUSION_HEADS': 16,
        'KSPACE_GNN_TYPE': 'transformer',
        
        # Training hyperparameters
        'LEARNING_RATE': 1e-3,
        'WEIGHT_DECAY': 1e-4,
        'MIXUP_ALPHA': 0.2,
        'CUTMIX_ALPHA': 1.0,
        'MASK_PROB': 0.1,
        'FOCAL_ALPHA': 1.0,
        'FOCAL_GAMMA': 2.0,
        'SCHEDULER_T0': 15,
        'SCHEDULER_T_MULT': 2,
        'SCHEDULER_ETA_MIN': 1e-6,
        
        # Pipeline settings
        'RUN_PRETRAINING': False,
        'RUN_HYPERPARAM_SEARCH': True,
        'RUN_BASELINES': True,
        'RUN_ENSEMBLE': True,
        'RUN_ANALYSIS': True,
        'N_TRIALS': 100,
        'EPOCHS_PER_TRIAL': 25,
        'ENSEMBLE_EPOCHS': 75,
        'FINAL_EPOCHS': 150,
        
        # Other settings
        'USE_WANDB': False,
        'SEED': 42
    }
    
    return config


def main():
    """Main function to run the crazy pipeline."""
    
    parser = argparse.ArgumentParser(description='Run the Crazy ML Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='crazy_pipeline_results', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--quick', action='store_true', help='Run quick version with fewer trials')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_crazy_config()
    
    # Modify config for quick run
    if args.quick:
        config['N_TRIALS'] = 20
        config['EPOCHS_PER_TRIAL'] = 10
        config['ENSEMBLE_EPOCHS'] = 30
        config['FINAL_EPOCHS'] = 50
        print("Running in quick mode with reduced epochs and trials")
    
    # Set random seed
    torch.manual_seed(config['SEED'])
    np.random.seed(config['SEED'])
    
    # Create pipeline
    pipeline = CrazyMLPipeline(config, args.device)
    
    # TODO: Load your actual data loaders here
    # For now, we'll create dummy loaders
    print("⚠️  WARNING: Using dummy data loaders. Replace with your actual data!")
    
    # Create dummy data loaders (replace with your actual data)
    class DummyDataset:
        def __init__(self, size=1000):
            self.size = size
            self.data = []
            for i in range(size):
                # Create dummy multimodal data
                sample = {
                    'crystal_x': torch.randn(50, 92),
                    'crystal_edge_index': torch.randint(0, 50, (2, 100)),
                    'kspace_x': torch.randn(30, 2),
                    'kspace_edge_index': torch.randint(0, 30, (2, 60)),
                    'scalar_features': torch.randn(200),
                    'decomposition_features': torch.randn(100)
                }
                self.data.append((sample, torch.randint(0, 2, (1,)).item()))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Create dummy datasets
    train_dataset = DummyDataset(800)
    val_dataset = DummyDataset(100)
    test_dataset = DummyDataset(100)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(train_loader, val_loader, test_loader, args.save_dir)
    
    print(f"\nPipeline completed! Check {args.save_dir} for results.")


if __name__ == "__main__":
    main() 