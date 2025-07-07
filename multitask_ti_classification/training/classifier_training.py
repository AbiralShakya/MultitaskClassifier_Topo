# classifier_training.py
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import random
from torch.optim.lr_scheduler import MultiStepLR

from src.model_with_topological_ml import EnhancedMultiModalMaterialClassifier
from helper.dataset import MaterialDataset
from helper import config

# --- Custom collate function (tuple-based) ---
def custom_collate_fn(batch_list):
    # each sample is (crystal_graph, asph_feats, kspace_graph, phys_feats_dict, topo_label, mag_label)
    crystal_graphs = [item[0] for item in batch_list]
    asph_feats    = [item[1] for item in batch_list]
    kspace_graphs = [item[2] for item in batch_list]
    phys_feats    = [item[3] for item in batch_list]
    topo_labels   = [item[4] for item in batch_list]

    batched_crystal = Batch.from_data_list(crystal_graphs)
    batched_kspace  = Batch.from_data_list(kspace_graphs)
    asph_feats      = torch.stack(asph_feats)
    topo_labels     = torch.stack(topo_labels)
    phys_collated   = {key: torch.stack([pf[key] for pf in phys_feats]) for key in phys_feats[0].keys()}

    return {
        'crystal_graph': batched_crystal,
        'asph_features': asph_feats,
        'kspace_graph':  batched_kspace,
        'kspace_physics_features': phys_collated,
        'topology_label': topo_labels
    }

# --- Helper to move batch to device ---
def move_batch_to_device(batch, device):
    batch['crystal_graph'] = batch['crystal_graph'].to(device)
    batch['asph_features'] = batch['asph_features'].to(device)
    batch['kspace_graph']  = batch['kspace_graph'].to(device)
    for key, val in batch['kspace_physics_features'].items():
        if torch.is_tensor(val):
            batch['kspace_physics_features'][key] = val.to(device)
    batch['topology_label'] = batch['topology_label'].to(device)
    return batch

# --- Metrics ---
def compute_metrics(preds, targs, num_classes, task_name, class_names=None):
    preds_np = preds.cpu().numpy()
    targs_np = targs.cpu().numpy()
    acc = accuracy_score(targs_np, preds_np)
    print(f"\n--- {task_name} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    report = classification_report(targs_np, preds_np,
                                   labels=list(range(num_classes)),
                                   target_names=class_names)
    cm = confusion_matrix(targs_np, preds_np,
                          labels=list(range(num_classes)))
    print(report)
    print("Confusion Matrix:")
    print(cm)
    return acc, report, cm

# --- Training Loop ---
def train():
    # Load dataset
    full_ds = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR
    )
    # Extract 3-way topology labels
    all_topo = np.array([full_ds[i][4].item() for i in range(len(full_ds))], dtype=np.int64)
    # Compute class weights avoiding zero division
    class_counts = np.bincount(all_topo, minlength=config.NUM_COMBINED_CLASSES)
    class_counts[class_counts==0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[all_topo]

    # Split indices
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    n = len(indices)
    train_idx = indices[:int(0.8*n)]
    val_idx   = indices[int(0.8*n):int(0.9*n)]
    test_idx  = indices[int(0.9*n):]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds   = torch.utils.data.Subset(full_ds, val_idx)
    test_ds  = torch.utils.data.Subset(full_ds, test_idx)

    # Weighted sampler
    train_sw = [sample_weights[i] for i in train_idx]
    train_sampler = WeightedRandomSampler(train_sw, len(train_ds), replacement=True)

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=train_sampler, num_workers=config.NUM_WORKERS,
                              collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              collate_fn=custom_collate_fn)

    # Model, optimizer, scheduler, loss
    device = config.DEVICE
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    model = EnhancedMultiModalMaterialClassifier(
        crystal_node_feature_dim=config.CRYSTAL_NODE_FEATURE_DIM,
        kspace_node_feature_dim=config.KSPACE_GRAPH_NODE_FEATURE_DIM,
        asph_feature_dim=config.ASPH_FEATURE_DIM,
        scalar_feature_dim=config.SCALAR_TOTAL_DIM,
        decomposition_feature_dim=config.BASE_DECOMPOSITION_FEATURE_DIM
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=config.LR_MILESTONES, gamma=config.LR_GAMMA)
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    aux_weight = config.AUXILIARY_WEIGHT

    # Training
    for epoch in range(1, config.NUM_EPOCHS+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model({
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph':  batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            })
            main_logits = outputs['combined_logits']
            aux_logits  = outputs['topology_logits_auxiliary']
            tgt_topo    = batch['topology_label'].long()

            loss_main = criterion(main_logits, tgt_topo)
            loss_aux  = aux_weight * criterion(aux_logits, (tgt_topo>0).long())
            loss = loss_main + loss_aux
            loss.backward()
            if config.MAX_GRAD_NORM>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        all_preds, all_targs = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                outputs = model({
                    'crystal_graph': batch['crystal_graph'],
                    'kspace_graph':  batch['kspace_graph'],
                    'asph_features': batch['asph_features'],
                    'kspace_physics_features': batch['kspace_physics_features']
                })
                preds = outputs['combined_logits'].argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targs.append(batch['topology_label'].cpu())
        all_preds = torch.cat(all_preds)
        all_targs = torch.cat(all_targs)
        compute_metrics(all_preds, all_targs, config.NUM_COMBINED_CLASSES,
                        "Combined Classification", class_names=["trivial","semimetal","topological_insulator"])

    # Test
    print("\n--- Test Evaluation ---")
    model.eval()
    all_preds, all_targs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model({
                'crystal_graph': batch['crystal_graph'],
                'kspace_graph':  batch['kspace_graph'],
                'asph_features': batch['asph_features'],
                'kspace_physics_features': batch['kspace_physics_features']
            })
            preds = outputs['combined_logits'].argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targs.append(batch['topology_label'].cpu())
    all_preds = torch.cat(all_preds)
    all_targs = torch.cat(all_targs)
    compute_metrics(all_preds, all_targs, config.NUM_COMBINED_CLASSES,
                    "Combined Classification Test", class_names=["trivial","semimetal","topological_insulator"])

if __name__ == '__main__':
    train()
