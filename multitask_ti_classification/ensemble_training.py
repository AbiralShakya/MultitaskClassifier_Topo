"""
Ensemble Training and Prediction - Train multiple models and aggregate predictions
for improved accuracy and robustness.
"""

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import VotingClassifier
import joblib
from tqdm import tqdm


class EnsembleModel:
    """Ensemble of multiple models with different configurations."""
    
    def __init__(self, model_configs: List[Dict], device: str = 'cuda'):
        self.model_configs = model_configs
        self.device = device
        self.models = []
        self.trainers = []
        self.model_paths = []
        
    def train_models(self, train_loader: DataLoader, val_loader: DataLoader, 
                    num_epochs: int, save_dir: str = 'ensemble_models'):
        """Train all models in the ensemble."""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training {len(self.model_configs)} models for ensemble...")
        
        for i, config in enumerate(self.model_configs):
            print(f"\n{'='*50}")
            print(f"Training Model {i+1}/{len(self.model_configs)}")
            print(f"Config: {config}")
            print(f"{'='*50}")
            
            # Set random seed for reproducibility
            seed = config.get('SEED', 42 + i)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Create model and trainer
            model = create_crazy_fusion_model(config)
            trainer = create_crazy_trainer(model, config, self.device)
            
            # Save model path
            model_path = os.path.join(save_dir, f'model_{i+1}.pth')
            self.model_paths.append(model_path)
            
            # Train model
            trainer.train(train_loader, val_loader, num_epochs, model_path)
            
            # Store model and trainer
            self.models.append(model)
            self.trainers.append(trainer)
            
            # Plot training curves
            trainer.plot_training_curves(os.path.join(save_dir, f'training_curves_{i+1}.png'))
            
            print(f"Model {i+1} training completed!")
        
        # Save ensemble metadata
        self.save_ensemble_metadata(save_dir)
        
        print(f"\nAll {len(self.model_configs)} models trained successfully!")
    
    def predict_ensemble(self, test_loader: DataLoader, method: str = 'soft_voting') -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions using different aggregation methods."""
        print(f"Making ensemble predictions using {method}...")
        
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for batch_data, _ in test_loader:
                    # Move to device
                    batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in batch_data.items()}
                    
                    # Forward pass
                    outputs = model(batch_data)
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())
            
            all_predictions.append(np.array(predictions))
            all_probabilities.append(np.array(probabilities))
        
        # Aggregate predictions
        if method == 'soft_voting':
            # Average probabilities
            avg_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = np.argmax(avg_probabilities, axis=1)
            ensemble_probabilities = avg_probabilities
        
        elif method == 'hard_voting':
            # Majority vote
            ensemble_predictions = []
            for i in range(len(all_predictions[0])):
                votes = [pred[i] for pred in all_predictions]
                ensemble_predictions.append(max(set(votes), key=votes.count))
            ensemble_predictions = np.array(ensemble_predictions)
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
        
        elif method == 'weighted_voting':
            # Weighted average based on validation accuracy
            weights = [trainer.best_val_acc for trainer in self.trainers]
            weights = np.array(weights) / sum(weights)
            
            weighted_probabilities = np.zeros_like(all_probabilities[0])
            for i, (prob, weight) in enumerate(zip(all_probabilities, weights)):
                weighted_probabilities += weight * prob
            
            ensemble_predictions = np.argmax(weighted_probabilities, axis=1)
            ensemble_probabilities = weighted_probabilities
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_predictions, ensemble_probabilities
    
    def evaluate_ensemble(self, test_loader: DataLoader, true_labels: List[int], 
                         method: str = 'soft_voting') -> Dict:
        """Evaluate ensemble performance."""
        predictions, probabilities = self.predict_ensemble(test_loader, method)
        
        # Calculate metrics with warning suppression
        accuracy = accuracy_score(true_labels, predictions)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted'
            )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def save_ensemble_metadata(self, save_dir: str):
        """Save ensemble metadata."""
        metadata = {
            'num_models': len(self.model_configs),
            'model_configs': self.model_configs,
            'model_paths': self.model_paths,
            'validation_accuracies': [trainer.best_val_acc for trainer in self.trainers]
        }
        
        with open(os.path.join(save_dir, 'ensemble_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_ensemble(self, save_dir: str):
        """Load trained ensemble models."""
        metadata_path = os.path.join(save_dir, 'ensemble_metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Ensemble metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.model_configs = metadata['model_configs']
        self.model_paths = metadata['model_paths']
        
        # Load models
        self.models = []
        self.trainers = []
        
        for i, (config, model_path) in enumerate(zip(self.model_configs, self.model_paths)):
            # Create model
            model = create_crazy_fusion_model(config)
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            # Create trainer for metadata
            trainer = create_crazy_trainer(model, config, self.device)
            trainer.best_val_acc = checkpoint['best_val_acc']
            
            self.models.append(model)
            self.trainers.append(trainer)
        
        print(f"Loaded ensemble with {len(self.models)} models")
    
    def plot_ensemble_results(self, metrics: Dict, save_path: str = 'ensemble_results.png'):
        """Plot ensemble results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Individual model accuracies
        val_accs = [trainer.best_val_acc for trainer in self.trainers]
        model_names = [f'Model {i+1}' for i in range(len(val_accs))]
        
        ax1.bar(model_names, val_accs)
        ax1.set_title('Individual Model Validation Accuracies')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Ensemble vs individual comparison
        ensemble_acc = metrics['accuracy'] * 100
        ax2.bar(['Ensemble'] + model_names, [ensemble_acc] + val_accs, 
                color=['red'] + ['blue'] * len(val_accs))
        ax2.set_title('Ensemble vs Individual Model Performance')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('Ensemble Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        
        # Metrics comparison
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1']]
        
        ax4.bar(metric_names, metric_values)
        ax4.set_title('Ensemble Performance Metrics')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_ensemble_configs() -> List[Dict]:
    """Create different model configurations for ensemble."""
    base_config = {
        'HIDDEN_DIM': 256,
        'FUSION_DIM': 512,
        'NUM_CLASSES': 2,
        'CRYSTAL_INPUT_DIM': 92,
        'KSPACE_INPUT_DIM': 2,
        'SCALAR_INPUT_DIM': 4763,
        'DECOMPOSITION_INPUT_DIM': 100,
        'K_EIGS': 64,
        'MIXUP_ALPHA': 0.2,
        'CUTMIX_ALPHA': 1.0,
        'MASK_PROB': 0.1,
        'FOCAL_ALPHA': 1.0,
        'FOCAL_GAMMA': 2.0,
        'LEARNING_RATE': 1e-3,
        'WEIGHT_DECAY': 1e-4,
        'SCHEDULER_T0': 10,
        'SCHEDULER_T_MULT': 2,
        'SCHEDULER_ETA_MIN': 1e-6,
        'USE_WANDB': False
    }
    
    configs = []
    
    # Config 1: All modalities, transformer GNN
    config1 = base_config.copy()
    config1.update({
        'SEED': 42,
        'USE_CRYSTAL': True,
        'USE_KSPACE': True,
        'USE_SCALAR': True,
        'USE_DECOMPOSITION': True,
        'USE_SPECTRAL': True,
        'KSPACE_GNN_TYPE': 'transformer',
        'CRYSTAL_LAYERS': 4,
        'KSPACE_LAYERS': 3,
        'SCALAR_BLOCKS': 3,
        'FUSION_BLOCKS': 3,
        'FUSION_HEADS': 8
    })
    configs.append(config1)
    
    # Config 2: All modalities, GAT GNN
    config2 = base_config.copy()
    config2.update({
        'SEED': 123,
        'USE_CRYSTAL': True,
        'USE_KSPACE': True,
        'USE_SCALAR': True,
        'USE_DECOMPOSITION': True,
        'USE_SPECTRAL': True,
        'KSPACE_GNN_TYPE': 'gat',
        'CRYSTAL_LAYERS': 3,
        'KSPACE_LAYERS': 4,
        'SCALAR_BLOCKS': 4,
        'FUSION_BLOCKS': 4,
        'FUSION_HEADS': 12
    })
    configs.append(config2)
    
    # Config 3: All modalities, GCN GNN
    config3 = base_config.copy()
    config3.update({
        'SEED': 456,
        'USE_CRYSTAL': True,
        'USE_KSPACE': True,
        'USE_SCALAR': True,
        'USE_DECOMPOSITION': True,
        'USE_SPECTRAL': True,
        'KSPACE_GNN_TYPE': 'gcn',
        'CRYSTAL_LAYERS': 5,
        'KSPACE_LAYERS': 2,
        'SCALAR_BLOCKS': 2,
        'FUSION_BLOCKS': 2,
        'FUSION_HEADS': 4
    })
    configs.append(config3)
    
    # Config 4: Crystal + K-space only
    config4 = base_config.copy()
    config4.update({
        'SEED': 789,
        'USE_CRYSTAL': True,
        'USE_KSPACE': True,
        'USE_SCALAR': False,
        'USE_DECOMPOSITION': False,
        'USE_SPECTRAL': False,
        'KSPACE_GNN_TYPE': 'sage',
        'CRYSTAL_LAYERS': 4,
        'KSPACE_LAYERS': 3,
        'FUSION_BLOCKS': 3,
        'FUSION_HEADS': 8
    })
    configs.append(config4)
    
    # Config 5: Scalar + Decomposition only
    config5 = base_config.copy()
    config5.update({
        'SEED': 101112,
        'USE_CRYSTAL': False,
        'USE_KSPACE': False,
        'USE_SCALAR': True,
        'USE_DECOMPOSITION': True,
        'USE_SPECTRAL': False,
        'SCALAR_BLOCKS': 5,
        'FUSION_BLOCKS': 2,
        'FUSION_HEADS': 6
    })
    configs.append(config5)
    
    return configs


def train_ensemble_pipeline(train_loader: DataLoader, val_loader: DataLoader, 
                           test_loader: DataLoader, true_labels: List[int],
                           num_epochs: int = 50, save_dir: str = 'ensemble_models'):
    """Complete ensemble training and evaluation pipeline."""
    
    # Create ensemble configurations
    configs = create_ensemble_configs()
    
    # Create ensemble
    ensemble = EnsembleModel(configs)
    
    # Train all models
    ensemble.train_models(train_loader, val_loader, num_epochs, save_dir)
    
    # Evaluate ensemble with different methods
    methods = ['soft_voting', 'hard_voting', 'weighted_voting']
    results = {}
    
    for method in methods:
        print(f"\nEvaluating ensemble with {method}...")
        metrics = ensemble.evaluate_ensemble(test_loader, true_labels, method)
        results[method] = metrics
        
        print(f"{method.upper()} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
    
    # Plot results
    best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
    ensemble.plot_ensemble_results(results[best_method], 
                                 os.path.join(save_dir, 'ensemble_results.png'))
    
    # Save results
    with open(os.path.join(save_dir, 'ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEnsemble training completed!")
    print(f"Best method: {best_method}")
    print(f"Best accuracy: {results[best_method]['accuracy']:.4f}")
    
    return ensemble, results


if __name__ == "__main__":
    # Test ensemble configurations
    configs = create_ensemble_configs()
    print(f"Created {len(configs)} ensemble configurations:")
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}:")
        print(f"  Seed: {config.get('SEED', 'N/A')}")
        print(f"  K-space GNN: {config.get('KSPACE_GNN_TYPE', 'N/A')}")
        print(f"  Modalities: Crystal={config.get('USE_CRYSTAL', False)}, "
              f"K-space={config.get('USE_KSPACE', False)}, "
              f"Scalar={config.get('USE_SCALAR', False)}, "
              f"Decomposition={config.get('USE_DECOMPOSITION', False)}, "
              f"Spectral={config.get('USE_SPECTRAL', False)}")
    
    # Create test ensemble
    ensemble = EnsembleModel(configs[:2])  # Just first 2 configs for testing
    print(f"\nTest ensemble created with {len(ensemble.model_configs)} models") 