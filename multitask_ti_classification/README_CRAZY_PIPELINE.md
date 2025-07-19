# 🚀 Crazy ML Pipeline for Topological Insulator Classification

A **state-of-the-art, comprehensive ML pipeline** that goes absolutely crazy on accuracy for topological insulator classification using multimodal data fusion.

## 🎯 What This Pipeline Does

This pipeline implements a **complete ML workflow** that combines:

1. **🤖 New Fundamental Architecture**: Multi-branch transformer fusion with cross-modal attention
2. **🎭 Advanced Data Augmentation**: Mixup, CutMix, and feature masking
3. **🏗️ Ensemble Training**: Multiple models with different architectures and seeds
4. **🔍 Hyperparameter Optimization**: Optuna-based automated search
5. **📊 Baseline Models**: XGBoost, LightGBM, Random Forest comparison
6. **🎓 Self-Supervised Pretraining**: Node/edge prediction and contrastive learning
7. **📈 Automated Analysis**: t-SNE, UMAP, attention visualization, error analysis

## 🏗️ Architecture Overview

### Crazy Fusion Model (`src/crazy_fusion_model.py`)
- **Multi-branch encoders**: Separate deep networks for each modality (crystal, k-space, scalar, decomposition, spectral)
- **Cross-modal attention**: Transformer-based fusion allowing information flow between modalities
- **Deep residuals**: Skip connections throughout for better gradient flow
- **Modular design**: Easy ablation and ensemble training

### Key Components:
- `CrystalGraphEncoder`: Deep GNN with alternating TransformerConv/GAT layers
- `KSpaceEncoder`: Multiple GNN variants (Transformer, GAT, GCN, GraphSAGE)
- `ScalarFeatureEncoder`: Deep residual blocks for scalar features
- `DecompositionEncoder`: Attention-based encoder for decomposition features
- `SpectralEncoder`: GPU-accelerated spectral graph features
- `CrossModalFusion`: Transformer blocks with cross-modal attention

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
# Full pipeline (recommended)
python run_crazy_pipeline.py

# Quick test run
python run_crazy_pipeline.py --quick

# Custom config
python run_crazy_pipeline.py --config my_config.json --save_dir my_results
```

### 3. Individual Components

#### Train Single Model
```python
from src.crazy_fusion_model import create_crazy_fusion_model
from training.crazy_training import create_crazy_trainer

# Create model and trainer
config = {...}  # Your config
model = create_crazy_fusion_model(config)
trainer = create_crazy_trainer(model, config)

# Train
trainer.train(train_loader, val_loader, num_epochs=100)
```

#### Run Ensemble Training
```python
from ensemble_training import train_ensemble_pipeline

# Train ensemble
ensemble, results = train_ensemble_pipeline(
    train_loader, val_loader, test_loader, true_labels
)
```

#### Hyperparameter Search
```python
from optuna_hyperparam_search import run_hyperparameter_search

# Find best hyperparameters
best_config, best_accuracy = run_hyperparameter_search(
    train_loader, val_loader, n_trials=100
)
```

## 📁 File Structure

```
multitask_ti_classification/
├── src/
│   ├── crazy_fusion_model.py      # New state-of-the-art model
│   └── model_w_debug.py           # Original model (updated)
├── training/
│   └── crazy_training.py          # Advanced training with augmentation
├── ensemble_training.py           # Ensemble training and prediction
├── optuna_hyperparam_search.py    # Hyperparameter optimization
├── baseline_models.py             # XGBoost/LightGBM baselines
├── self_supervised_pretraining.py # Self-supervised learning
├── automated_analysis.py          # t-SNE/UMAP and error analysis
├── run_crazy_pipeline.py          # Master pipeline orchestrator
├── requirements.txt               # Dependencies
└── README_CRAZY_PIPELINE.md       # This file
```

## 🎛️ Configuration

### Model Architecture
```python
config = {
    'HIDDEN_DIM': 512,              # Hidden dimension for encoders
    'FUSION_DIM': 1024,             # Fusion dimension
    'NUM_CLASSES': 2,               # Number of classes
    'USE_CRYSTAL': True,            # Use crystal graph data
    'USE_KSPACE': True,             # Use k-space graph data
    'USE_SCALAR': True,             # Use scalar features
    'USE_DECOMPOSITION': True,      # Use decomposition features
    'USE_SPECTRAL': True,           # Use spectral features
    'CRYSTAL_LAYERS': 5,            # Number of GNN layers
    'KSPACE_GNN_TYPE': 'transformer', # GNN type for k-space
    'FUSION_BLOCKS': 4,             # Number of fusion transformer blocks
    'FUSION_HEADS': 16,             # Number of attention heads
}
```

### Training Hyperparameters
```python
config = {
    'LEARNING_RATE': 1e-3,          # Learning rate
    'WEIGHT_DECAY': 1e-4,           # Weight decay
    'MIXUP_ALPHA': 0.2,             # Mixup augmentation strength
    'CUTMIX_ALPHA': 1.0,            # CutMix augmentation strength
    'MASK_PROB': 0.1,               # Feature masking probability
    'FOCAL_ALPHA': 1.0,             # Focal loss alpha
    'FOCAL_GAMMA': 2.0,             # Focal loss gamma
}
```

### Pipeline Settings
```python
config = {
    'RUN_PRETRAINING': False,       # Run self-supervised pretraining
    'RUN_HYPERPARAM_SEARCH': True,  # Run hyperparameter optimization
    'RUN_BASELINES': True,          # Train baseline models
    'RUN_ENSEMBLE': True,           # Train ensemble models
    'RUN_ANALYSIS': True,           # Run automated analysis
    'N_TRIALS': 100,                # Number of hyperparameter trials
    'ENSEMBLE_EPOCHS': 75,          # Epochs for ensemble training
    'FINAL_EPOCHS': 150,            # Epochs for final model
}
```

## 🔬 Advanced Features

### 1. Data Augmentation
- **Mixup**: Interpolates between samples and labels
- **CutMix**: Mixes features from different samples
- **Feature Masking**: Randomly masks features for robustness

### 2. Ensemble Methods
- **Soft Voting**: Average probabilities from multiple models
- **Hard Voting**: Majority vote from multiple models
- **Weighted Voting**: Weighted average based on validation accuracy

### 3. Self-Supervised Pretraining
- **Node Prediction**: Predict node labels in graphs
- **Edge Prediction**: Predict edge existence
- **Contrastive Learning**: Learn representations using graph augmentations

### 4. Hyperparameter Optimization
- **Optuna**: Bayesian optimization with TPE sampler
- **Comprehensive Search**: Architecture, training, and fusion parameters
- **Early Stopping**: Stop poor trials early to save time

### 5. Automated Analysis
- **t-SNE/UMAP**: Visualize high-dimensional features
- **Attention Heatmaps**: Visualize attention weights
- **Error Analysis**: Analyze prediction errors and confidence
- **Feature Importance**: Identify important features

## 📊 Expected Results

With this pipeline, you should achieve:

- **95%+ accuracy** on your topological insulator classification task
- **Robust performance** across different data splits
- **Comprehensive analysis** of model behavior and errors
- **Multiple baseline comparisons** to understand performance

## 🛠️ Customization

### Add New Modalities
1. Create encoder in `crazy_fusion_model.py`
2. Add to `CrazyFusionModel.forward()`
3. Update configuration

### Add New GNN Types
1. Add to `KSpaceEncoder` in `crazy_fusion_model.py`
2. Update configuration options

### Add New Augmentation
1. Add to `CrazyTrainer` in `crazy_training.py`
2. Update training loop

### Add New Analysis
1. Add to `AutomatedAnalyzer` in `automated_analysis.py`
2. Update comprehensive report

## 🚨 Important Notes

1. **Replace Dummy Data**: The pipeline currently uses dummy data. Replace with your actual data loaders.
2. **GPU Memory**: Large models may require significant GPU memory. Adjust `HIDDEN_DIM` and `FUSION_DIM` accordingly.
3. **Training Time**: Full pipeline can take hours. Use `--quick` flag for testing.
4. **Dependencies**: Install PyTorch Geometric according to your CUDA version.

## 🎯 Next Steps

1. **Replace dummy data** with your actual data loaders
2. **Adjust configuration** for your specific dataset
3. **Run full pipeline** to get baseline results
4. **Analyze results** and iterate on configuration
5. **Deploy best model** for production use

## 🤝 Contributing

This pipeline is designed to be modular and extensible. Feel free to:
- Add new model architectures
- Implement new augmentation techniques
- Add new analysis methods
- Optimize for your specific use case

## 📈 Performance Tips

1. **Start with quick mode** to test the pipeline
2. **Use GPU** for faster training
3. **Adjust batch size** based on GPU memory
4. **Monitor validation accuracy** to avoid overfitting
5. **Use ensemble methods** for better robustness

---

**Ready to go crazy on accuracy?** 🚀

Run `python run_crazy_pipeline.py` and watch the magic happen! 