import itertools
import random
import importlib
import shutil
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Hyperparameter grid
learning_rates = [1e-3, 5e-4, 1e-4]
dropouts = [0.05, 0.1, 0.2]
fusion_mlps = [[512,128], [1024,512,128]]
gnn_layers = [4, 8]
gnn_types = ['transformer', 'gcn', 'gat', 'sage']

# Random samples
random_trials = 3

# Results
results = []

# Helper to set config values
def set_config(lr, dropout, fusion, gnn_layers, gnn_type):
    import helper.config as config
    config.LEARNING_RATE = lr
    config.DROPOUT_RATE = dropout
    config.FUSION_HIDDEN_DIMS = fusion
    config.GNN_NUM_LAYERS = gnn_layers
    config.KSPACE_GNN_TYPE = gnn_type
    # Optionally set ablation flags here
    # config.USE_CRYSTAL = ...
    # config.USE_KSPACE = ...
    # config.USE_SCALAR = ...
    # config.USE_DECOMPOSITION = ...
    # Save config for reproducibility
    with open('last_config.txt', 'w') as f:
        f.write(f'lr={lr}, dropout={dropout}, fusion={fusion}, gnn_layers={gnn_layers}, gnn_type={gnn_type}\n')

# Grid search
grid = list(itertools.product(learning_rates, dropouts, fusion_mlps, gnn_layers, gnn_types))
random.shuffle(grid)

# Add random samples
for _ in range(random_trials):
    grid.append((random.choice(learning_rates),
                 random.choice(dropouts),
                 random.choice(fusion_mlps),
                 random.choice(gnn_layers),
                 random.choice(gnn_types)))

for i, (lr, dropout, fusion, gnn_layer, gnn_type) in enumerate(tqdm(grid, desc='Hyperparam search')):
    print(f'\n=== Trial {i+1}/{len(grid)} ===')
    set_config(lr, dropout, fusion, gnn_layer, gnn_type)
    # Reload config in training module
    importlib.reload(importlib.import_module('helper.config'))
    # Remove old checkpoints
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    # Run training (assume main_training_loop logs/returns best val acc/loss)
    from training.classifier_training import main_training_loop
    try:
        best_val_acc, best_val_loss = main_training_loop(max_epochs=3, quick_mode=True)  # You may need to add these args
    except Exception as e:
        print(f'Error in trial {i+1}: {e}')
        best_val_acc, best_val_loss = 0, 999
    results.append({'lr': lr, 'dropout': dropout, 'fusion': fusion, 'gnn_layers': gnn_layer, 'gnn_type': gnn_type,
                    'val_acc': best_val_acc, 'val_loss': best_val_loss})
    # Save best model for this run
    if os.path.exists('checkpoints/best_model.pt'):
        shutil.copy('checkpoints/best_model.pt', f'best_model_trial_{i+1}.pt')

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('hyperparam_search_results.csv', index=False)
print('All results saved to hyperparam_search_results.csv')

# Print best config
best_row = results_df.sort_values('val_acc', ascending=False).iloc[0]
print('\nBest config:')
print(best_row) 