import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from torch.optim.lr_scheduler import MultiStepLR
from model import HybridTopoClassifier
from helper.dataset import MaterialDataset
from helper import config

def custom_collate_fn(batch):
    """
    Custom collate function to handle mixed data types:
    - PyTorch Geometric Data objects (crystal_graph, kspace_graph)
    - Regular tensors (asph_features, kspace_physics_features)
    - Labels
    """
    # Separate the components
    crystal_graphs = [item[0] for item in batch]
    asph_features = [item[1] for item in batch]
    kspace_graphs = [item[2] for item in batch]
    kspace_physics_features = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    
    # Batch the PyG Data objects
    batched_crystal_graph = Batch.from_data_list(crystal_graphs)
    batched_kspace_graph = Batch.from_data_list(kspace_graphs)
    
    # Stack the regular tensors
    asph_features = torch.stack(asph_features)
    labels = torch.stack(labels)
    
    # Handle kspace_physics_features dictionary
    batched_kspace_physics = {}
    for key in kspace_physics_features[0].keys():
        batched_kspace_physics[key] = torch.stack([item[key] for item in kspace_physics_features])
    
    return batched_crystal_graph, asph_features, batched_kspace_graph, batched_kspace_physics, labels

# --- Focal Loss for handling class imbalance ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Metrics ---
def compute_metrics(predictions, targets, num_classes, task_name, class_names=None):
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    acc = accuracy_score(targets_np, preds_np)
    labels = list(range(num_classes))
    if class_names is None:
        class_names = [f"Class {i}" for i in labels]
    report = classification_report(targets_np, preds_np, output_dict=True, zero_division='warn', labels=labels, target_names=class_names)
    cm = confusion_matrix(targets_np, preds_np, labels=labels)
    print(f"\n--- {task_name} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Classification Report:\n{json.dumps(report, indent=2)}")
    print(f"Confusion Matrix:\n{cm}")
    return acc, report, cm

# --- Training Loop ---
def train():
    # Dataset and DataLoader
    full_dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR
    )
    
    # Get all labels for computing class weights and sampler
    all_labels = []
    for i in range(len(full_dataset)):
        _, _, _, _, label = full_dataset[i]
        all_labels.append(label.item())
    all_labels = np.array(all_labels, dtype=np.int64)
    
    # Compute class weights for aggressive oversampling
    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / class_counts
    
    # Create sample weights for WeightedRandomSampler
    sample_weights = class_weights[all_labels]
    
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    train_split = int(0.8 * len(indices))
    val_split = int(0.9 * len(indices))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create weighted sampler for training set
    train_labels = [all_labels[i] for i in train_indices]
    train_sample_weights = class_weights[train_labels].tolist()
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler, num_workers=config.NUM_WORKERS, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=custom_collate_fn)

    # Model
    model = HybridTopoClassifier(
        cgcnn_node_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        cgcnn_edge_dim=config.CRYSTAL_EDGE_FEATURE_DIM,
        asph_dim=config.ASPH_FEATURE_DIM,
        kspace_node_dim=config.KSPACE_NODE_FEATURE_DIM,
        kspace_edge_dim=config.KSPACE_EDGE_FEATURE_DIM,
        kspace_physics_dim=config.DECOMPOSITION_FEATURE_DIM + config.BAND_GAP_SCALAR_DIM + config.DOS_FEATURE_DIM + config.FERMI_FEATURE_DIM
    ).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    
    # Use CrossEntropyLoss with class weights for better handling of class imbalance
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            crystal_graph, asph_features, kspace_graph, kspace_physics_features, label = batch
            crystal_graph = crystal_graph.to(config.DEVICE)
            asph_features = asph_features.to(config.DEVICE)
            kspace_graph = kspace_graph.to(config.DEVICE)
            for k in kspace_physics_features:
                kspace_physics_features[k] = kspace_physics_features[k].to(config.DEVICE)
            label = label.to(config.DEVICE)
            optimizer.zero_grad()
            
            # Get all logits from the improved model
            main_logits, cgcnn_logits, asph_logits, kspace_logits = model(crystal_graph, asph_features, kspace_graph, kspace_physics_features)
            
            # Compute losses for each modality with better weighting
            main_loss = criterion(main_logits, label)
            cgcnn_loss = criterion(cgcnn_logits, label)
            asph_loss = criterion(asph_logits, label)
            kspace_loss = criterion(kspace_logits, label)
            
            # Progressive auxiliary loss weighting (reduce over time)
            aux_weight = max(0.1, 0.5 * (1 - epoch / config.NUM_EPOCHS))
            total_batch_loss = main_loss + aux_weight * (cgcnn_loss + asph_loss + kspace_loss)
            
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                crystal_graph, asph_features, kspace_graph, kspace_physics_features, label = batch
                crystal_graph = crystal_graph.to(config.DEVICE)
                asph_features = asph_features.to(config.DEVICE)
                kspace_graph = kspace_graph.to(config.DEVICE)
                for k in kspace_physics_features:
                    kspace_physics_features[k] = kspace_physics_features[k].to(config.DEVICE)
                label = label.to(config.DEVICE)
                
                # Get main logits for validation
                main_logits, _, _, _ = model(crystal_graph, asph_features, kspace_graph, kspace_physics_features)
                preds = torch.argmax(main_logits, dim=1)
                all_preds.append(preds)
                all_targets.append(label)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        compute_metrics(all_preds, all_targets, config.NUM_TOPOLOGY_CLASSES, "Topology Classification")

    # Final Test Evaluation
    print("\n--- Final Test Set Evaluation ---")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            crystal_graph, asph_features, kspace_graph, kspace_physics_features, label = batch
            crystal_graph = crystal_graph.to(config.DEVICE)
            asph_features = asph_features.to(config.DEVICE)
            kspace_graph = kspace_graph.to(config.DEVICE)
            for k in kspace_physics_features:
                kspace_physics_features[k] = kspace_physics_features[k].to(config.DEVICE)
            label = label.to(config.DEVICE)
            
            # Get main logits for test evaluation
            main_logits, _, _, _ = model(crystal_graph, asph_features, kspace_graph, kspace_physics_features)
            preds = torch.argmax(main_logits, dim=1)
            all_preds.append(preds)
            all_targets.append(label)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    compute_metrics(all_preds, all_targets, config.NUM_TOPOLOGY_CLASSES, "Topology Classification")

if __name__ == '__main__':
    train()