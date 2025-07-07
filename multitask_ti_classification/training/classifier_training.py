# classifier_training.py
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import random
from pathlib import Path
from torch.optim.lr_scheduler import MultiStepLR

# Local imports
from src.model_with_topological_ml import EnhancedMultiModalMaterialClassifier
from helper.dataset import MaterialDataset
from helper import config

# --- Custom collate function to batch heterogeneous data ---
def custom_collate_fn(batch_list):
    """
    Custom collate to batch: crystal_graph, asph_features, kspace_graph,
    kspace_physics_features (dict), topology_label, magnetism_label, other metadata.
    """
    if not batch_list:
        return {}
    # collect
    crystal_graphs = [d['crystal_graph'] for d in batch_list]
    asph_feats    = [d['asph_features']     for d in batch_list]
    kspace_graphs = [d['kspace_graph']       for d in batch_list]
    phys_feats    = [d['kspace_physics_features'] for d in batch_list]
    topo_labels   = [d['topology_label']     for d in batch_list]
    mag_labels    = [d['magnetism_label']    for d in batch_list]
    # batch PyG graphs
    batched_crystal = Batch.from_data_list(crystal_graphs)
    batched_kspace  = Batch.from_data_list(kspace_graphs)
    # stack tensors
    asph_feats = torch.stack(asph_feats)
    topo_labels = torch.stack(topo_labels)
    mag_labels  = torch.stack(mag_labels)
    # collate physics dict
    phys_collated = {}
    for key in phys_feats[0].keys():
        phys_collated[key] = torch.stack([pf[key] for pf in phys_feats])
    return {
        'crystal_graph': batched_crystal,
        'asph_features': asph_feats,
        'kspace_graph':  batched_kspace,
        'kspace_physics_features': phys_collated,
        'topology_label': topo_labels,
        'magnetism_label': mag_labels
    }

# --- Focal Loss (optional) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1-pt)**self.gamma * ce
        if self.reduction=='mean': return loss.mean()
        if self.reduction=='sum':  return loss.sum()
        return loss

# --- Metrics ---
def compute_metrics(preds, targs, num_classes, task_name, class_names=None):
    preds_np = preds.cpu().numpy()
    targs_np = targs.cpu().numpy()
    acc = accuracy_score(targs_np, preds_np)
    print(f"\n--- {task_name} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    report = classification_report(targs_np, preds_np, labels=list(range(num_classes)), target_names=class_names)
    cm = confusion_matrix(targs_np, preds_np, labels=list(range(num_classes)))
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    return acc, report, cm

# --- Training Loop ---
def train():
    # 1) Load full dataset and compute sampling weights for 3 classes
    full_ds = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR
    )
    # extract labels
    all_topo = []
    for i in range(len(full_ds)):
        item = full_ds[i]
        all_topo.append(item['topology_label'].item())
    all_topo = np.array(all_topo, dtype=np.int64)
    class_counts = np.bincount(all_topo, minlength=config.NUM_COMBINED_CLASSES)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[all_topo]

    # train/val/test split
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    n = len(indices)
    train_end = int(0.8*n)
    val_end   = int(0.9*n)
    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds   = torch.utils.data.Subset(full_ds, val_idx)
    test_ds  = torch.utils.data.Subset(full_ds, test_idx)

    # Weighted sampler
    train_sw = [sample_weights[i] for i in train_idx]
    train_sampler = WeightedRandomSampler(train_sw, num_samples=len(train_ds), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=train_sampler, num_workers=config.NUM_WORKERS,
                              collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              collate_fn=custom_collate_fn)

    # 2) Model, optimizer, scheduler, loss
    device = config.DEVICE
    model = EnhancedMultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.BASE_DECOMPOSITION_FEATURE_DIM
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=config.LR_MILESTONES, gamma=config.LR_GAMMA)
    # loss with class weights
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    aux_weight = config.AUXILIARY_WEIGHT

    # 3) Training epochs
    for epoch in range(1, config.NUM_EPOCHS+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # move to device
            batch = {k: (v.to(device) if torch.is_tensor(v) else {kk:kv.to(device) for kk,kv in v.items()})
                     for k,v in batch.items()}
            optimizer.zero_grad()
            # forward
            outputs = model({
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph':  batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'scalar_features': batch['scalar_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            })
            # logits
            main_logits = outputs['combined_logits']
            topo_logits = outputs['topology_logits_primary']
            topo_aux   = outputs.get('topology_logits_auxiliary')
            mag_logits = outputs['magnetism_logits_aux']
            # targets
            tgt_topo = batch['topology_label'].long()
            tgt_mag  = batch['magnetism_label'].long()
            # losses
            loss_main = criterion(main_logits, tgt_topo)
            loss_topo = aux_weight * criterion(topo_logits, (tgt_topo>0).long())
            if topo_aux is not None:
                loss_topo += aux_weight * criterion(topo_aux, (tgt_topo>0).long())
            loss_mag  = aux_weight * criterion(mag_logits, tgt_mag)
            loss = loss_main + loss_topo + loss_mag
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} - Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_targs = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else {kk:kv.to(device) for kk,kv in v.items()})
                         for k,v in batch.items()}
                outs = model({
                    'crystal_graph': batch['crystal_graph'],
                    'kspace_graph':  batch['kspace_graph'],
                    'asph_features': batch['asph_features'],
                    'scalar_features': batch['scalar_features'],
                    'kspace_physics_features': batch['kspace_physics_features']
                })
                preds = outs['combined_logits'].argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targs.append(batch['topology_label'].cpu())
        all_preds = torch.cat(all_preds)
        all_targs = torch.cat(all_targs)
        compute_metrics(all_preds, all_targs, config.NUM_COMBINED_CLASSES,
                        "Combined Classification",
                        class_names=["trivial","semimetal","topological_insulator"])

    # 4) Final Test Evaluation
    print("\n--- Final Test Set Evaluation ---")
    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else {kk:kv.to(device) for kk,kv in v.items()})
                     for k,v in batch.items()}
            outs = model({
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph':  batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'scalar_features': batch['scalar_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            })
            preds = outs['combined_logits'].argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targs.append(batch['topology_label'].cpu())
    all_preds = torch.cat(all_preds)
    all_targs = torch.cat(all_targs)
    compute_metrics(all_preds, all_targs, config.NUM_COMBINED_CLASSES,
                    "Combined Classification Test",
                    class_names=["trivial","semimetal","topological_insulator"])

if __name__ == '__main__':
    train()
