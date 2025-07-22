#!/usr/bin/env python3
"""
Integration script to apply all enhancements to the existing training pipeline
"""

import os
import sys
from pathlib import Path

def update_training_file():
    """Update the training file to use enhanced model"""
    
    training_file = Path("training/classifier_training.py")
    
    if not training_file.exists():
        print(f"Training file not found: {training_file}")
        return False
    
    # Read the current training file
    with open(training_file, 'r') as f:
        content = f.read()
    
    # Check if already updated
    if "EnhancedIntegratedMaterialClassifier" in content:
        print("Training file already updated with enhanced model")
        return True
    
    # Replace the old model class definition
    old_class_def = "class EnhancedMultiModalMaterialClassifier(nn.Module):"
    new_class_def = """# Use the enhanced integrated model
from src.enhanced_integrated_model import EnhancedIntegratedMaterialClassifier

# Backward compatibility alias
EnhancedMultiModalMaterialClassifier = EnhancedIntegratedMaterialClassifier"""
    
    if old_class_def in content:
        # Find the end of the old class definition
        class_start = content.find(old_class_def)
        if class_start != -1:
            # Find the next class or function definition
            next_def = content.find("\nclass ", class_start + 1)
            if next_def == -1:
                next_def = content.find("\ndef ", class_start + 1)
            
            if next_def != -1:
                # Replace the old class with the import
                content = content[:class_start] + new_class_def + "\n\n" + content[next_def:]
            else:
                # If no next definition found, replace until end
                content = content[:class_start] + new_class_def + "\n"
    
    # Write back the updated content
    with open(training_file, 'w') as f:
        f.write(content)
    
    print("✅ Training file updated with enhanced model")
    return True

def create_enhanced_training_script():
    """Create a new enhanced training script"""
    
    enhanced_script = Path("training/enhanced_classifier_training.py")
    
    script_content = '''#!/usr/bin/env python3
"""
Enhanced classifier training with all improvements from the Nature paper:
1. Enhanced atomic features (65D)
2. Voronoi graph construction  
3. Multi-scale attention
4. Topological consistency loss
5. Persistent homology integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

# Enhanced model and components
from src.enhanced_integrated_model import create_enhanced_model
from helper.enhanced_topological_loss import EnhancedTopologicalLoss, FocalLoss
from helper.dataset import MaterialDataset, custom_collate_fn
import helper.config as config


def main():
    """Enhanced training with all improvements"""
    print("🚀 Starting Enhanced Topological Material Classification Training")
    print("=" * 70)
    
    # Device setup
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = MaterialDataset(
        master_index_path=config.MASTER_INDEX_PATH,
        kspace_graphs_base_dir=config.KSPACE_GRAPHS_DIR,
        data_root_dir=config.DATA_DIR,
        dos_fermi_dir=config.DOS_FERMI_DIR,
        preload=getattr(config, 'PRELOAD_DATASET', True)
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Stratified split
    try:
        labels = [dataset[i]['topology_label'].item() for i in range(len(dataset))]
        train_indices, temp_indices = train_test_split(
            range(len(dataset)), test_size=0.3, random_state=42, stratify=labels
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42,
            stratify=[labels[i] for i in temp_indices]
        )
    except Exception as e:
        print(f"Stratified split failed: {e}, using random split")
        train_indices, temp_indices = train_test_split(
            range(len(dataset)), test_size=0.3, random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42
        )
    
    print(f"Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Create data loaders
    train_loader = PyGDataLoader(
        Subset(dataset, train_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = PyGDataLoader(
        Subset(dataset, val_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    test_loader = PyGDataLoader(
        Subset(dataset, test_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=config.NUM_WORKERS
    )
    
    # Create enhanced model
    print("Creating enhanced model...")
    model = create_enhanced_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Enhanced optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-4),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config.PATIENCE, factor=0.7, min_lr=1e-6
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\\nStarting training...")
    print("=" * 70)
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif hasattr(batch[key], 'to'):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if isinstance(batch[key][sub_key], torch.Tensor):
                            batch[key][sub_key] = batch[key][sub_key].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = model.compute_loss(outputs, batch['topology_label'])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            
            # Statistics
            train_losses.append(loss.item())
            preds = outputs['logits'].argmax(dim=1)
            train_correct += (preds == batch['topology_label']).sum().item()
            train_total += len(batch['topology_label'])
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                    elif hasattr(batch[key], 'to'):
                        batch[key] = batch[key].to(device)
                    elif isinstance(batch[key], dict):
                        for sub_key in batch[key]:
                            if isinstance(batch[key][sub_key], torch.Tensor):
                                batch[key][sub_key] = batch[key][sub_key].to(device)
                
                outputs = model(batch)
                loss = model.compute_loss(outputs, batch['topology_label'])
                
                val_losses.append(loss.item())
                preds = outputs['logits'].argmax(dim=1)
                val_correct += (preds == batch['topology_label']).sum().item()
                val_total += len(batch['topology_label'])
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_enhanced_model.pt')
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Final evaluation
    print("\\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    # Load best model
    model.load_state_dict(torch.load('best_enhanced_model.pt'))
    model.eval()
    
    test_preds = []
    test_targets = []
    test_confidences = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif hasattr(batch[key], 'to'):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if isinstance(batch[key][sub_key], torch.Tensor):
                            batch[key][sub_key] = batch[key][sub_key].to(device)
            
            # Get predictions with confidence
            pred_results = model.predict_with_confidence(batch)
            
            test_preds.extend(pred_results['predictions'].cpu().numpy())
            test_targets.extend(batch['topology_label'].cpu().numpy())
            test_confidences.extend(pred_results['confidence'].cpu().numpy())
    
    # Calculate final metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    conf_matrix = confusion_matrix(test_targets, test_preds)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Average Confidence: {np.mean(test_confidences):.4f}")
    print("\\nConfusion Matrix:")
    print(conf_matrix)
    print("\\nClassification Report:")
    print(classification_report(test_targets, test_preds, digits=4))
    
    print("\\n🎉 Enhanced training completed successfully!")


if __name__ == "__main__":
    main()
'''
    
    with open(enhanced_script, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Enhanced training script created: {enhanced_script}")
    return True

def main():
    """Main integration function"""
    print("🔧 Integrating Enhanced Topological Material Classification")
    print("=" * 60)
    
    print("\\n1. Updating training file...")
    update_training_file()
    
    print("\\n2. Creating enhanced training script...")
    create_enhanced_training_script()
    
    print("\\n✅ Integration completed successfully!")
    print("\\nNext steps:")
    print("1. Run: python training/enhanced_classifier_training.py")
    print("2. Or use existing training with enhanced model automatically")
    print("\\nEnhancements included:")
    print("• 65D enhanced atomic features (vs 3D)")
    print("• Voronoi graph construction")
    print("• Multi-scale attention networks")
    print("• Topological consistency loss")
    print("• Persistent homology integration")
    print("• Focal loss for class imbalance")
    print("• Confidence estimation")

if __name__ == "__main__":
    main()