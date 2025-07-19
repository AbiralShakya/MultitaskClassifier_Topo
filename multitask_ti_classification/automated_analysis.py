"""
Automated Analysis - Comprehensive visualization and error analysis
including t-SNE/UMAP, attention heatmaps, feature importance, and error analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import json
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.crazy_fusion_model import create_crazy_fusion_model
from helper.config import *
from tqdm import tqdm


class AutomatedAnalyzer:
    """Comprehensive automated analysis for model evaluation."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.features = {}
        self.attention_weights = {}
        self.predictions = {}
        self.true_labels = {}
        
    def extract_features_and_predictions(self, data_loader: DataLoader, 
                                       split_name: str = 'test') -> Dict:
        """Extract features, attention weights, and predictions from model."""
        
        print(f"Extracting features and predictions for {split_name} split...")
        
        self.model.eval()
        all_features = []
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch_data, labels in tqdm(data_loader, desc=f"Processing {split_name}"):
                # Move to device
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Extract features from fusion layer (if available)
                if hasattr(self.model, 'fusion'):
                    # Hook to get fusion features
                    fusion_features = self._get_fusion_features(batch_data)
                    all_features.extend(fusion_features.cpu().numpy())
                
                # Extract attention weights (if available)
                if hasattr(self.model, 'get_attention_weights'):
                    attention_weights = self.model.get_attention_weights()
                    if attention_weights:
                        all_attention_weights.append(attention_weights)
        
        # Store results
        self.features[split_name] = np.array(all_features) if all_features else None
        self.predictions[split_name] = np.array(all_predictions)
        self.true_labels[split_name] = np.array(all_labels)
        self.attention_weights[split_name] = all_attention_weights if all_attention_weights else None
        
        results = {
            'features': self.features[split_name],
            'predictions': self.predictions[split_name],
            'true_labels': self.true_labels[split_name],
            'attention_weights': self.attention_weights[split_name]
        }
        
        return results
    
    def _get_fusion_features(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from fusion layer."""
        # This is a simplified version - you might need to modify based on your model architecture
        modality_features = {}
        
        # Extract features from each modality
        if hasattr(self.model, 'encoders'):
            for name, encoder in self.model.encoders.items():
                if name == 'crystal' and 'crystal_x' in batch_data:
                    features = encoder(batch_data['crystal_x'], 
                                     batch_data['crystal_edge_index'],
                                     batch_data.get('crystal_batch'))
                    modality_features[name] = features
                
                elif name == 'kspace' and 'kspace_x' in batch_data:
                    features = encoder(batch_data['kspace_x'], 
                                     batch_data['kspace_edge_index'],
                                     batch_data.get('kspace_batch'))
                    modality_features[name] = features
                
                elif name == 'scalar' and 'scalar_features' in batch_data:
                    features = encoder(batch_data['scalar_features'])
                    modality_features[name] = features
                
                elif name == 'decomposition' and 'decomposition_features' in batch_data:
                    features = encoder(batch_data['decomposition_features'])
                    modality_features[name] = features
        
        # Concatenate all features
        if modality_features:
            return torch.cat(list(modality_features.values()), dim=1)
        else:
            return torch.zeros(batch_data.get('scalar_features', torch.zeros(1, 100)).size(0), 512)
    
    def perform_tsne_analysis(self, split_name: str = 'test', 
                            save_path: str = 'tsne_analysis.png') -> np.ndarray:
        """Perform t-SNE analysis on extracted features."""
        
        if self.features[split_name] is None:
            print("No features available for t-SNE analysis")
            return None
        
        print(f"Performing t-SNE analysis for {split_name} split...")
        
        # Reduce dimensionality with PCA first
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(self.features[split_name])
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features_pca)
        
        # Plot t-SNE results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color by true labels
        scatter1 = ax1.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                              c=self.true_labels[split_name], cmap='viridis', alpha=0.7)
        ax1.set_title(f't-SNE: True Labels ({split_name})')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=ax1)
        
        # Color by predictions
        scatter2 = ax2.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                              c=self.predictions[split_name], cmap='viridis', alpha=0.7)
        ax2.set_title(f't-SNE: Predictions ({split_name})')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return features_tsne
    
    def perform_umap_analysis(self, split_name: str = 'test', 
                            save_path: str = 'umap_analysis.png') -> np.ndarray:
        """Perform UMAP analysis on extracted features."""
        
        if self.features[split_name] is None:
            print("No features available for UMAP analysis")
            return None
        
        print(f"Performing UMAP analysis for {split_name} split...")
        
        # Perform UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        features_umap = reducer.fit_transform(self.features[split_name])
        
        # Plot UMAP results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color by true labels
        scatter1 = ax1.scatter(features_umap[:, 0], features_umap[:, 1], 
                              c=self.true_labels[split_name], cmap='viridis', alpha=0.7)
        ax1.set_title(f'UMAP: True Labels ({split_name})')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        plt.colorbar(scatter1, ax=ax1)
        
        # Color by predictions
        scatter2 = ax2.scatter(features_umap[:, 0], features_umap[:, 1], 
                              c=self.predictions[split_name], cmap='viridis', alpha=0.7)
        ax2.set_title(f'UMAP: Predictions ({split_name})')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return features_umap
    
    def analyze_errors(self, split_name: str = 'test', 
                      save_path: str = 'error_analysis.png') -> Dict:
        """Analyze prediction errors."""
        
        print(f"Analyzing errors for {split_name} split...")
        
        true_labels = self.true_labels[split_name]
        predictions = self.predictions[split_name]
        
        # Find errors
        errors = predictions != true_labels
        error_indices = np.where(errors)[0]
        
        # Calculate metrics
        accuracy = np.mean(predictions == true_labels)
        error_rate = 1 - accuracy
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Plot error analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Error distribution
        ax2.hist(true_labels[errors], bins=len(np.unique(true_labels)), alpha=0.7, label='Errors')
        ax2.hist(true_labels, bins=len(np.unique(true_labels)), alpha=0.7, label='All')
        ax2.set_title('Error Distribution by Class')
        ax2.set_xlabel('True Class')
        ax2.set_ylabel('Count')
        ax2.legend()
        
        # Prediction confidence for errors vs correct
        if hasattr(self, 'probabilities') and self.probabilities.get(split_name) is not None:
            probabilities = np.array(self.probabilities[split_name])
            max_probs = np.max(probabilities, axis=1)
            
            ax3.hist(max_probs[errors], bins=20, alpha=0.7, label='Errors', density=True)
            ax3.hist(max_probs[~errors], bins=20, alpha=0.7, label='Correct', density=True)
            ax3.set_title('Prediction Confidence Distribution')
            ax3.set_xlabel('Max Probability')
            ax3.set_ylabel('Density')
            ax3.legend()
        
        # Error rate by class
        unique_labels = np.unique(true_labels)
        error_rates_by_class = []
        for label in unique_labels:
            class_mask = true_labels == label
            class_error_rate = np.mean(predictions[class_mask] != true_labels[class_mask])
            error_rates_by_class.append(class_error_rate)
        
        ax4.bar(unique_labels, error_rates_by_class)
        ax4.set_title('Error Rate by Class')
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Error Rate')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create error analysis summary
        error_analysis = {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'num_errors': len(error_indices),
            'error_indices': error_indices.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'error_rates_by_class': dict(zip(unique_labels, error_rates_by_class))
        }
        
        return error_analysis
    
    def visualize_attention(self, split_name: str = 'test', 
                           save_path: str = 'attention_heatmap.png'):
        """Visualize attention weights."""
        
        if self.attention_weights[split_name] is None:
            print("No attention weights available for visualization")
            return
        
        print(f"Visualizing attention weights for {split_name} split...")
        
        # Aggregate attention weights
        all_attention = []
        for batch_attention in self.attention_weights[split_name]:
            if isinstance(batch_attention, dict):
                for key, value in batch_attention.items():
                    if isinstance(value, torch.Tensor):
                        all_attention.append(value.cpu().numpy())
        
        if not all_attention:
            print("No valid attention weights found")
            return
        
        # Average attention weights
        avg_attention = np.mean(all_attention, axis=0)
        
        # Plot attention heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if avg_attention.ndim == 2:
            sns.heatmap(avg_attention, cmap='viridis', ax=ax)
            ax.set_title('Average Attention Weights')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
        else:
            # For 1D attention weights
            ax.plot(avg_attention)
            ax.set_title('Average Attention Weights')
            ax.set_xlabel('Position')
            ax.set_ylabel('Attention Weight')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_feature_importance(self, split_name: str = 'test', 
                                 save_path: str = 'feature_importance.png'):
        """Analyze feature importance using correlation with predictions."""
        
        if self.features[split_name] is None:
            print("No features available for feature importance analysis")
            return
        
        print(f"Analyzing feature importance for {split_name} split...")
        
        features = self.features[split_name]
        predictions = self.predictions[split_name]
        
        # Calculate correlation between features and predictions
        correlations = []
        for i in range(features.shape[1]):
            correlation = np.corrcoef(features[:, i], predictions)[0, 1]
            correlations.append(abs(correlation))
        
        # Sort features by importance
        feature_importance = np.array(correlations)
        top_indices = np.argsort(feature_importance)[-20:]  # Top 20 features
        
        # Plot feature importance
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.barh(range(len(top_indices)), feature_importance[top_indices])
        ax.set_title('Feature Importance (Top 20)')
        ax.set_xlabel('Absolute Correlation with Predictions')
        ax.set_ylabel('Feature Index')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance
    
    def create_comprehensive_report(self, save_dir: str = 'analysis_results') -> Dict:
        """Create comprehensive analysis report."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating comprehensive analysis report...")
        
        report = {}
        
        # Perform all analyses
        for split_name in self.features.keys():
            split_dir = os.path.join(save_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            split_report = {}
            
            # t-SNE analysis
            tsne_features = self.perform_tsne_analysis(
                split_name, os.path.join(split_dir, 'tsne_analysis.png')
            )
            split_report['tsne_features'] = tsne_features.tolist() if tsne_features is not None else None
            
            # UMAP analysis
            umap_features = self.perform_umap_analysis(
                split_name, os.path.join(split_dir, 'umap_analysis.png')
            )
            split_report['umap_features'] = umap_features.tolist() if umap_features is not None else None
            
            # Error analysis
            error_analysis = self.analyze_errors(
                split_name, os.path.join(split_dir, 'error_analysis.png')
            )
            split_report['error_analysis'] = error_analysis
            
            # Attention visualization
            self.visualize_attention(
                split_name, os.path.join(split_dir, 'attention_heatmap.png')
            )
            
            # Feature importance
            feature_importance = self.analyze_feature_importance(
                split_name, os.path.join(split_dir, 'feature_importance.png')
            )
            split_report['feature_importance'] = feature_importance.tolist() if feature_importance is not None else None
            
            report[split_name] = split_report
        
        # Save comprehensive report
        with open(os.path.join(save_dir, 'comprehensive_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary plots
        self._create_summary_plots(save_dir)
        
        print(f"Comprehensive analysis completed! Results saved to {save_dir}")
        
        return report
    
    def _create_summary_plots(self, save_dir: str):
        """Create summary plots comparing different splits."""
        
        # Accuracy comparison across splits
        accuracies = {}
        for split_name in self.features.keys():
            true_labels = self.true_labels[split_name]
            predictions = self.predictions[split_name]
            accuracies[split_name] = np.mean(predictions == true_labels)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.bar(accuracies.keys(), accuracies.values())
        ax.set_title('Accuracy Comparison Across Splits')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        for i, (split, acc) in enumerate(accuracies.items()):
            ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


def run_automated_analysis(model: nn.Module, train_loader: DataLoader, 
                          val_loader: DataLoader, test_loader: DataLoader,
                          save_dir: str = 'automated_analysis') -> Dict:
    """Run complete automated analysis pipeline."""
    
    # Create analyzer
    analyzer = AutomatedAnalyzer(model)
    
    # Extract features and predictions for all splits
    train_results = analyzer.extract_features_and_predictions(train_loader, 'train')
    val_results = analyzer.extract_features_and_predictions(val_loader, 'val')
    test_results = analyzer.extract_features_and_predictions(test_loader, 'test')
    
    # Create comprehensive report
    report = analyzer.create_comprehensive_report(save_dir)
    
    return report


if __name__ == "__main__":
    # Test automated analyzer
    print("Testing automated analysis...")
    
    # Create dummy model
    config = {
        'HIDDEN_DIM': 256,
        'FUSION_DIM': 512,
        'NUM_CLASSES': 2,
        'USE_CRYSTAL': True,
        'USE_KSPACE': True,
        'USE_SCALAR': True,
        'USE_DECOMPOSITION': True,
        'USE_SPECTRAL': True
    }
    
    model = create_crazy_fusion_model(config)
    analyzer = AutomatedAnalyzer(model)
    
    print("Automated analyzer initialized successfully!") 