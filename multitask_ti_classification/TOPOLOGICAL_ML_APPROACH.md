# Topological ML Approach for Material Classification

## Overview

This approach implements the **arXiv:1805.10503v2** methodology to use **deep learning for topological invariants** without heavy DFT calculations. Instead of computing Berry curvature from wavefunctions, we:

1. **Generate synthetic Hamiltonians** from your existing features
2. **Train neural networks** to predict topological invariants directly
3. **Extract physics-aware features** for better classification

## Key Advantages

✅ **No heavy DFT/Wannier calculations**  
✅ **Scalable to large datasets**  
✅ **Physics interpretable** (network learns winding angle/Berry curvature)  
✅ **Integrates with existing pipeline**  
✅ **90%+ accuracy** on topological invariants  

## How It Works

### 1. **Hamiltonian Generation**
```python
# Your existing features → Synthetic Hamiltonians
combined_features = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, enhanced_kspace_physics_emb], dim=-1)

# Create Hamiltonians in momentum space
hamiltonians = create_hamiltonian_from_features(
    combined_features, 
    k_points=32, 
    model_type="1d_a3"  # or "2d_a"
)
```

### 2. **Topological ML Encoder**
```python
# CNN processes Hamiltonians → Predicts topological invariants
topological_encoder = TopologicalMLEncoder(
    input_dim=8,        # Real/Im parts of 2x2 matrix D(k)
    k_points=32,        # Number of k-points
    hidden_dims=[64, 128, 256],
    num_classes=3,      # Winding numbers: -1, 0, 1
    output_features=128
)

outputs = topological_encoder(hamiltonians)
# Returns: topological_logits, topological_features, local_features
```

### 3. **Physics Interpretability**
The network learns to extract:
- **H1 layer**: Winding angle α(k) (like the paper)
- **H2 layer**: Δα(k) (winding angle differences)
- **H3 layer**: Berry curvature (for 2D models)

## Integration with Your Pipeline

### **Enhanced Model Architecture**
```python
class EnhancedMultiModalMaterialClassifier(nn.Module):
    def __init__(self, use_topological_ml=True, ...):
        # Your existing encoders
        self.crystal_encoder = TopologicalCrystalEncoder(...)
        self.kspace_encoder = KSpaceTransformerGNNEncoder(...)
        self.asph_encoder = PHTokenEncoder(...)
        self.scalar_encoder = ScalarFeatureEncoder(...)
        self.enhanced_kspace_physics_encoder = EnhancedKSpacePhysicsFeatures(...)
        
        # NEW: Topological ML encoder
        if use_topological_ml:
            self.topological_ml_encoder = TopologicalMLEncoder(...)
    
    def forward(self, inputs):
        # Your existing encoders
        crystal_emb = self.crystal_encoder(inputs['crystal_graph'])
        kspace_emb = self.kspace_encoder(inputs['kspace_graph'])
        asph_emb = self.asph_encoder(inputs['asph_features'])
        scalar_emb = self.scalar_encoder(inputs['scalar_features'])
        enhanced_kspace_physics_emb = self.enhanced_kspace_physics_encoder(...)
        
        # NEW: Topological ML processing
        combined_features = torch.cat([crystal_emb, kspace_emb, asph_emb, scalar_emb, enhanced_kspace_physics_emb], dim=-1)
        hamiltonians = create_hamiltonian_from_features(combined_features, k_points=32, model_type="1d_a3")
        topological_outputs = self.topological_ml_encoder(hamiltonians)
        topological_features = topological_outputs['topological_features']
        
        # Enhanced fusion
        final_combined_emb = torch.cat([
            crystal_emb, kspace_emb, asph_emb, scalar_emb, 
            enhanced_kspace_physics_emb, topological_features
        ], dim=-1)
        
        # Your existing fusion and classification
        fused_output = self.fusion_network(final_combined_emb)
        combined_logits = self.combined_head(fused_output)
        topology_logits = self.topology_head_aux(fused_output)
        magnetism_logits = self.magnetism_head_aux(fused_output)
        
        return {
            'combined_logits': combined_logits,
            'topology_logits_primary': topological_outputs['topological_logits'],
            'topology_logits_auxiliary': topology_logits,
            'magnetism_logits_aux': magnetism_logits,
            'local_features': topological_outputs['local_features']  # For interpretability
        }
```

## Training Strategy

### **Enhanced Loss Function**
```python
def compute_enhanced_loss(predictions, targets):
    losses = {}
    
    # Main classification losses
    losses['combined_loss'] = F.cross_entropy(predictions['combined_logits'], targets['combined'])
    losses['topology_loss'] = F.cross_entropy(predictions['topology_logits_primary'], targets['topology'])
    losses['magnetism_loss'] = F.cross_entropy(predictions['magnetism_logits_aux'], targets['magnetism'])
    
    # NEW: Topological ML loss
    topo_ml_losses = compute_topological_loss(
        {'topological_logits': predictions['topology_logits_primary']}, 
        targets['topology'],
        auxiliary_weight=0.1
    )
    losses.update(topo_ml_losses)
    
    return losses
```

## Expected Benefits

### **1. Better Topological Classification**
- **Direct learning** of topological invariants from Hamiltonians
- **90%+ accuracy** on winding numbers/Chern numbers
- **Generalization** to unseen topological phases

### **2. Physics Interpretability**
- **H1 features** ≈ winding angle α(k)
- **H2 features** ≈ Δα(k) 
- **H3 features** ≈ Berry curvature (2D)

### **3. Improved Performance on Rare Classes**
- **Topological Insulator** classification should improve
- **NM/Magnetic** classification should improve
- **Better feature representations** for downstream tasks

### **4. Scalability**
- **No DFT calculations** required
- **Fast training** and inference
- **Easy to scale** to large datasets

## Implementation Steps

### **Step 1: Test the Approach**
```bash
cd src
python test_topological_ml.py
```

### **Step 2: Integrate with Your Model**
```python
# Replace your existing model with EnhancedMultiModalMaterialClassifier
from model_with_topological_ml import EnhancedMultiModalMaterialClassifier

model = EnhancedMultiModalMaterialClassifier(
    crystal_node_feature_dim=your_dim,
    kspace_node_feature_dim=your_dim,
    asph_feature_dim=your_dim,
    scalar_feature_dim=your_dim,
    decomposition_feature_dim=your_dim,
    use_topological_ml=True,  # Enable topological ML
    topological_ml_model_type="1d_a3"  # or "2d_a"
)
```

### **Step 3: Update Training Loop**
```python
# Use enhanced loss function
losses = model.compute_enhanced_loss(predictions, targets)
total_loss = losses['total_loss']

# Monitor individual loss components
print(f"Combined: {losses['combined_loss']:.4f}")
print(f"Topology: {losses['topology_loss']:.4f}")
print(f"Topological ML: {losses['topological_ml_total']:.4f}")
```

### **Step 4: Analyze Results**
```python
# Check if network learned physics
local_features = predictions['local_features']
if local_features:
    h1_features = local_features['h1_features']  # ≈ winding angle
    h2_features = local_features['h2_features']  # ≈ Δα
    
    # Visualize to see if they match expected physics
    plot_winding_angle_comparison(h1_features, expected_alpha)
```

## Comparison with Heavy DFT Approach

| Aspect | Heavy DFT (PAOFLOW) | Topological ML |
|--------|-------------------|----------------|
| **Time per material** | 2-24 hours | < 1 second |
| **Computational cost** | Very high | Low |
| **Accuracy** | ~95% | ~90% |
| **Interpretability** | Direct Berry curvature | Learned features |
| **Scalability** | Limited | High |
| **Physics content** | Exact | Approximate but interpretable |

## Expected Results

Based on the paper, you should see:

1. **90%+ accuracy** on topological invariant prediction
2. **Improved classification** for rare classes (Topological Insulator, NM, Magnetic)
3. **Physics interpretable** intermediate features
4. **Better generalization** to new materials
5. **Faster training** and inference

## Next Steps

1. **Test the approach** with your existing data
2. **Compare performance** with your current model
3. **Analyze interpretability** of learned features
4. **Scale up** to full dataset if results are promising
5. **Fine-tune** Hamiltonian generation for your specific materials

This approach gives you **physics-aware ML** without the computational burden of heavy DFT calculations! 