"""
Self-Supervised Pretraining - Pretrain GNN encoders using unlabeled data
with node prediction, edge prediction, and contrastive learning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import json
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.crazy_fusion_model import CrystalGraphEncoder, KSpaceEncoder
from helper.config import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class NodePredictionTask(nn.Module):
    """Node prediction task for self-supervised learning."""
    
    def __init__(self, encoder: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode graph
        node_embeddings = self.encoder(x, edge_index, batch)
        
        # Predict node labels
        node_predictions = self.node_predictor(node_embeddings)
        
        return node_predictions


class EdgePredictionTask(nn.Module):
    """Edge prediction task for self-supervised learning."""
    
    def __init__(self, encoder: nn.Module, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode graph
        node_embeddings = self.encoder(x, edge_index, batch)
        
        # Get edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        
        # Predict edge existence
        edge_predictions = self.edge_predictor(edge_embeddings)
        
        return edge_predictions.squeeze(-1)


class ContrastiveLearningTask(nn.Module):
    """Contrastive learning task using graph augmentations."""
    
    def __init__(self, encoder: nn.Module, hidden_dim: int, temperature: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x1: torch.Tensor, edge_index1: torch.Tensor,
                x2: torch.Tensor, edge_index2: torch.Tensor,
                batch1: Optional[torch.Tensor] = None,
                batch2: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Encode both views
        z1 = self.encoder(x1, edge_index1, batch1)
        z2 = self.encoder(x2, edge_index2, batch2)
        
        # Project to contrastive space
        h1 = self.projector(z1)
        h2 = self.projector(z2)
        
        # Normalize
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(h1, h2.T) / self.temperature
        
        return sim_matrix


class GraphAugmentation:
    """Graph augmentation techniques for contrastive learning."""
    
    def __init__(self, node_dropout: float = 0.1, edge_dropout: float = 0.1, 
                 feature_noise: float = 0.1):
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.feature_noise = feature_noise
    
    def augment_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations to graph."""
        
        # Feature noise
        if self.feature_noise > 0:
            noise = torch.randn_like(x) * self.feature_noise
            x = x + noise
        
        # Node dropout
        if self.node_dropout > 0:
            num_nodes = x.size(0)
            keep_nodes = torch.rand(num_nodes) > self.node_dropout
            keep_indices = torch.where(keep_nodes)[0]
            
            if len(keep_indices) > 0:
                x = x[keep_indices]
                # Update edge_index
                node_mapping = torch.zeros(num_nodes, dtype=torch.long)
                node_mapping[keep_indices] = torch.arange(len(keep_indices))
                edge_index = node_mapping[edge_index]
                # Remove edges with dropped nodes
                valid_edges = (edge_index[0] >= 0) & (edge_index[1] >= 0)
                edge_index = edge_index[:, valid_edges]
        
        # Edge dropout
        if self.edge_dropout > 0:
            num_edges = edge_index.size(1)
            keep_edges = torch.rand(num_edges) > self.edge_dropout
            edge_index = edge_index[:, keep_edges]
        
        return x, edge_index


class SelfSupervisedTrainer:
    """Trainer for self-supervised learning tasks."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train_node_prediction(self, train_loader: DataLoader, val_loader: DataLoader,
                            num_epochs: int, save_path: str = 'node_prediction_model.pth'):
        """Train node prediction task."""
        
        print("Training node prediction task...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_data, node_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                # Move to device
                x = batch_data['x'].to(self.device)
                edge_index = batch_data['edge_index'].to(self.device)
                batch = batch_data.get('batch', None)
                if batch is not None:
                    batch = batch.to(self.device)
                node_labels = node_labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(x, edge_index, batch)
                loss = F.cross_entropy(predictions, node_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate_node_prediction(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
        
        print(f"Node prediction training completed! Best val loss: {best_val_loss:.4f}")
    
    def train_edge_prediction(self, train_loader: DataLoader, val_loader: DataLoader,
                            num_epochs: int, save_path: str = 'edge_prediction_model.pth'):
        """Train edge prediction task."""
        
        print("Training edge prediction task...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_data, edge_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                # Move to device
                x = batch_data['x'].to(self.device)
                edge_index = batch_data['edge_index'].to(self.device)
                batch = batch_data.get('batch', None)
                if batch is not None:
                    batch = batch.to(self.device)
                edge_labels = edge_labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(x, edge_index, batch)
                loss = F.binary_cross_entropy_with_logits(predictions, edge_labels.float())
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate_edge_prediction(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
        
        print(f"Edge prediction training completed! Best val loss: {best_val_loss:.4f}")
    
    def train_contrastive_learning(self, train_loader: DataLoader, val_loader: DataLoader,
                                 num_epochs: int, save_path: str = 'contrastive_model.pth'):
        """Train contrastive learning task."""
        
        print("Training contrastive learning task...")
        
        augmentation = GraphAugmentation()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                # Move to device
                x = batch_data['x'].to(self.device)
                edge_index = batch_data['edge_index'].to(self.device)
                batch = batch_data.get('batch', None)
                if batch is not None:
                    batch = batch.to(self.device)
                
                # Create two augmented views
                x1, edge_index1 = augmentation.augment_graph(x, edge_index)
                x2, edge_index2 = augmentation.augment_graph(x, edge_index)
                
                # Forward pass
                self.optimizer.zero_grad()
                sim_matrix = self.model(x1, edge_index1, x2, edge_index2, batch, batch)
                
                # Contrastive loss
                labels = torch.arange(sim_matrix.size(0)).to(self.device)
                loss = F.cross_entropy(sim_matrix, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate_contrastive(val_loader, augmentation)
            
            # Update scheduler
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
        
        print(f"Contrastive learning training completed! Best val loss: {best_val_loss:.4f}")
    
    def _validate_node_prediction(self, val_loader: DataLoader) -> float:
        """Validate node prediction task."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data, node_labels in val_loader:
                x = batch_data['x'].to(self.device)
                edge_index = batch_data['edge_index'].to(self.device)
                batch = batch_data.get('batch', None)
                if batch is not None:
                    batch = batch.to(self.device)
                node_labels = node_labels.to(self.device)
                
                predictions = self.model(x, edge_index, batch)
                loss = F.cross_entropy(predictions, node_labels)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _validate_edge_prediction(self, val_loader: DataLoader) -> float:
        """Validate edge prediction task."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data, edge_labels in val_loader:
                x = batch_data['x'].to(self.device)
                edge_index = batch_data['edge_index'].to(self.device)
                batch = batch_data.get('batch', None)
                if batch is not None:
                    batch = batch.to(self.device)
                edge_labels = edge_labels.to(self.device)
                
                predictions = self.model(x, edge_index, batch)
                loss = F.binary_cross_entropy_with_logits(predictions, edge_labels.float())
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _validate_contrastive(self, val_loader: DataLoader, augmentation: GraphAugmentation) -> float:
        """Validate contrastive learning task."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data, _ in val_loader:
                x = batch_data['x'].to(self.device)
                edge_index = batch_data['edge_index'].to(self.device)
                batch = batch_data.get('batch', None)
                if batch is not None:
                    batch = batch.to(self.device)
                
                # Create two augmented views
                x1, edge_index1 = augmentation.augment_graph(x, edge_index)
                x2, edge_index2 = augmentation.augment_graph(x, edge_index)
                
                sim_matrix = self.model(x1, edge_index1, x2, edge_index2, batch, batch)
                labels = torch.arange(sim_matrix.size(0)).to(self.device)
                loss = F.cross_entropy(sim_matrix, labels)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)


def create_pretraining_tasks(config: dict) -> Dict[str, nn.Module]:
    """Create self-supervised pretraining tasks."""
    
    tasks = {}
    
    # Node prediction task
    if config.get('USE_NODE_PREDICTION', True):
        crystal_encoder = CrystalGraphEncoder(
            input_dim=config.get('CRYSTAL_INPUT_DIM', 92),
            hidden_dim=config.get('HIDDEN_DIM', 256),
            output_dim=config.get('HIDDEN_DIM', 256),
            num_layers=config.get('CRYSTAL_LAYERS', 4)
        )
        
        node_task = NodePredictionTask(
            crystal_encoder,
            config.get('HIDDEN_DIM', 256),
            config.get('NUM_NODE_CLASSES', 10)
        )
        tasks['node_prediction'] = node_task
    
    # Edge prediction task
    if config.get('USE_EDGE_PREDICTION', True):
        kspace_encoder = KSpaceEncoder(
            input_dim=config.get('KSPACE_INPUT_DIM', 2),
            hidden_dim=config.get('HIDDEN_DIM', 256),
            output_dim=config.get('HIDDEN_DIM', 256),
            gnn_type=config.get('KSPACE_GNN_TYPE', 'transformer'),
            num_layers=config.get('KSPACE_LAYERS', 3)
        )
        
        edge_task = EdgePredictionTask(
            kspace_encoder,
            config.get('HIDDEN_DIM', 256)
        )
        tasks['edge_prediction'] = edge_task
    
    # Contrastive learning task
    if config.get('USE_CONTRASTIVE', True):
        contrastive_encoder = CrystalGraphEncoder(
            input_dim=config.get('CRYSTAL_INPUT_DIM', 92),
            hidden_dim=config.get('HIDDEN_DIM', 256),
            output_dim=config.get('HIDDEN_DIM', 256),
            num_layers=config.get('CRYSTAL_LAYERS', 4)
        )
        
        contrastive_task = ContrastiveLearningTask(
            contrastive_encoder,
            config.get('HIDDEN_DIM', 256),
            temperature=config.get('CONTRASTIVE_TEMPERATURE', 0.1)
        )
        tasks['contrastive'] = contrastive_task
    
    return tasks


def run_pretraining_pipeline(train_loader: DataLoader, val_loader: DataLoader,
                           config: dict, save_dir: str = 'pretrained_models') -> Dict[str, str]:
    """Run complete self-supervised pretraining pipeline."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create pretraining tasks
    tasks = create_pretraining_tasks(config)
    
    # Train each task
    model_paths = {}
    
    for task_name, task_model in tasks.items():
        print(f"\n{'='*50}")
        print(f"Training {task_name} task...")
        print(f"{'='*50}")
        
        # Create trainer
        trainer = SelfSupervisedTrainer(task_model)
        
        # Train task
        save_path = os.path.join(save_dir, f'{task_name}_model.pth')
        
        if task_name == 'node_prediction':
            trainer.train_node_prediction(train_loader, val_loader, 
                                        config.get('PRETRAINING_EPOCHS', 50), save_path)
        elif task_name == 'edge_prediction':
            trainer.train_edge_prediction(train_loader, val_loader, 
                                        config.get('PRETRAINING_EPOCHS', 50), save_path)
        elif task_name == 'contrastive':
            trainer.train_contrastive_learning(train_loader, val_loader, 
                                             config.get('PRETRAINING_EPOCHS', 50), save_path)
        
        model_paths[task_name] = save_path
    
    # Save pretraining config
    with open(os.path.join(save_dir, 'pretraining_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nPretraining completed! Models saved to {save_dir}")
    
    return model_paths


if __name__ == "__main__":
    # Test pretraining tasks
    print("Testing self-supervised pretraining...")
    
    config = {
        'HIDDEN_DIM': 256,
        'CRYSTAL_INPUT_DIM': 92,
        'KSPACE_INPUT_DIM': 2,
        'CRYSTAL_LAYERS': 4,
        'KSPACE_LAYERS': 3,
        'KSPACE_GNN_TYPE': 'transformer',
        'NUM_NODE_CLASSES': 10,
        'CONTRASTIVE_TEMPERATURE': 0.1,
        'USE_NODE_PREDICTION': True,
        'USE_EDGE_PREDICTION': True,
        'USE_CONTRASTIVE': True,
        'PRETRAINING_EPOCHS': 50
    }
    
    tasks = create_pretraining_tasks(config)
    print(f"Created {len(tasks)} pretraining tasks:")
    for task_name in tasks.keys():
        print(f"  - {task_name}")
    
    print("Self-supervised pretraining setup completed!") 