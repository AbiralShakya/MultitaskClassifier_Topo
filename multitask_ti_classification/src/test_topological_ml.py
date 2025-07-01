#!/usr/bin/env python3
"""
Test script for Topological ML Encoder integration.
Shows how to use the arXiv:1805.10503v2 approach in your existing pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

from topological_ml_encoder import (
    TopologicalMLEncoder, 
    TopologicalMLEncoder2D,
    create_hamiltonian_from_features,
    compute_topological_loss
)

def test_1d_topological_encoder():
    """Test 1D topological encoder (AIII class, winding number)."""
    print("Testing 1D Topological ML Encoder...")
    
    # Create encoder
    encoder = TopologicalMLEncoder(
        input_dim=8,  # Real/Im parts of 2x2 matrix D(k)
        k_points=32,  # Number of k-points
        hidden_dims=[64, 128, 256],
        num_classes=3,  # Winding numbers: -1, 0, 1
        output_features=128,
        extract_local_features=True
    )
    
    # Create synthetic data
    batch_size = 16
    features = torch.randn(batch_size, 10)  # Random features from your existing encoders
    
    # Create Hamiltonians from features
    hamiltonians = create_hamiltonian_from_features(
        features, k_points=32, model_type="1d_a3"
    )
    
    print(f"Input shape: {hamiltonians.shape}")  # Should be (16, 8, 32)
    
    # Forward pass
    outputs = encoder(hamiltonians)
    
    print("Output keys:", outputs.keys())
    print(f"Topological logits shape: {outputs['topological_logits'].shape}")
    print(f"Topological features shape: {outputs['topological_features'].shape}")
    
    if outputs['local_features']:
        print(f"H1 features shape: {outputs['local_features']['h1_features'].shape}")
        print(f"H2 features shape: {outputs['local_features']['h2_features'].shape}")
    
    # Test loss computation
    targets = torch.randint(0, 3, (batch_size,))  # Random topological labels
    loss_dict = compute_topological_loss(outputs, targets)
    
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Main loss: {loss_dict['main_loss'].item():.4f}")
    print(f"Feature loss: {loss_dict['feature_loss'].item():.4f}")
    
    return encoder, outputs

def test_2d_topological_encoder():
    """Test 2D topological encoder (A class, Chern number)."""
    print("\nTesting 2D Topological ML Encoder...")
    
    # Create encoder
    encoder = TopologicalMLEncoder2D(
        input_dim=3,  # hx, hy, hz components of H(k) = h(k)·σ
        k_grid=8,     # 8x8 k-grid
        hidden_dims=[32, 64, 128],
        num_classes=5,  # Chern numbers: -2, -1, 0, 1, 2
        output_features=128,
        extract_berry_curvature=True
    )
    
    # Create synthetic data
    batch_size = 16
    features = torch.randn(batch_size, 10)  # Random features
    
    # Create 2D Hamiltonians from features
    hamiltonians = create_hamiltonian_from_features(
        features, k_points=64, model_type="2d_a"  # 64 = 8*8
    )
    
    print(f"Input shape: {hamiltonians.shape}")  # Should be (16, 3, 8, 8)
    
    # Forward pass
    outputs = encoder(hamiltonians)
    
    print("Output keys:", outputs.keys())
    print(f"Chern logits shape: {outputs['chern_logits'].shape}")
    print(f"Topological features shape: {outputs['topological_features'].shape}")
    
    if outputs['berry_curvature'] is not None:
        print(f"Berry curvature shape: {outputs['berry_curvature'].shape}")
    
    # Test loss computation
    targets = torch.randint(0, 5, (batch_size,))  # Random Chern number labels
    loss_dict = compute_topological_loss(outputs, targets)
    
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Main loss: {loss_dict['main_loss'].item():.4f}")
    print(f"Feature loss: {loss_dict['feature_loss'].item():.4f}")
    
    return encoder, outputs

def integrate_with_existing_pipeline():
    """Show how to integrate with your existing MultiModalMaterialClassifier."""
    print("\nIntegration with Existing Pipeline...")
    
    # Simulate your existing encoders
    batch_size = 8
    
    # Your existing embeddings (from your current model)
    crystal_emb = torch.randn(batch_size, 128)
    kspace_emb = torch.randn(batch_size, 128)
    asph_emb = torch.randn(batch_size, 128)
    scalar_emb = torch.randn(batch_size, 128)
    enhanced_kspace_physics_emb = torch.randn(batch_size, 128)
    
    # Combine existing features
    combined_features = torch.cat([
        crystal_emb, kspace_emb, asph_emb, scalar_emb, enhanced_kspace_physics_emb
    ], dim=-1)
    
    print(f"Combined features shape: {combined_features.shape}")
    
    # Create topological ML encoder
    topological_encoder = TopologicalMLEncoder(
        input_dim=8,
        k_points=32,
        hidden_dims=[64, 128, 256],
        num_classes=3,
        output_features=128
    )
    
    # Create Hamiltonians from your existing features
    hamiltonians = create_hamiltonian_from_features(
        combined_features, k_points=32, model_type="1d_a3"
    )
    
    # Get topological predictions
    topological_outputs = topological_encoder(hamiltonians)
    
    # Extract topological features for fusion
    topological_features = topological_outputs['topological_features']
    
    print(f"Topological features shape: {topological_features.shape}")
    
    # Now you can concatenate with your existing features
    final_combined_features = torch.cat([
        combined_features, topological_features
    ], dim=-1)
    
    print(f"Final combined features shape: {final_combined_features.shape}")
    
    return topological_encoder, topological_outputs, final_combined_features

def visualize_hamiltonians():
    """Visualize the generated Hamiltonians."""
    print("\nVisualizing Generated Hamiltonians...")
    
    # Create sample features
    features = torch.randn(1, 10)
    
    # Create 1D Hamiltonian
    hamiltonian_1d = create_hamiltonian_from_features(
        features, k_points=32, model_type="1d_a3"
    )
    
    # Create 2D Hamiltonian
    hamiltonian_2d = create_hamiltonian_from_features(
        features, k_points=64, model_type="2d_a"
    )
    
    # Plot 1D Hamiltonian components
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for i in range(4):
        plt.plot(hamiltonian_1d[0, i].numpy(), label=f'D{i//2+1}{i%2+1} (Re)' if i < 2 else f'D{i//2+1}{i%2+1} (Im)')
    plt.title('1D Hamiltonian - Real Parts')
    plt.xlabel('k-point')
    plt.ylabel('Value')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    for i in range(4, 8):
        plt.plot(hamiltonian_1d[0, i].numpy(), label=f'D{i//2+1}{i%2+1} (Re)' if i < 6 else f'D{i//2+1}{i%2+1} (Im)')
    plt.title('1D Hamiltonian - Imaginary Parts')
    plt.xlabel('k-point')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot 2D Hamiltonian
    plt.subplot(1, 3, 3)
    hx = hamiltonian_2d[0, 0].numpy()
    plt.imshow(hx, cmap='RdBu_r', aspect='equal')
    plt.title('2D Hamiltonian - hx component')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('hamiltonian_visualization.png', dpi=150, bbox_inches='tight')
    print("Hamiltonian visualization saved as 'hamiltonian_visualization.png'")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Topological ML Encoder Test Suite")
    print("Inspired by arXiv:1805.10503v2")
    print("=" * 60)
    
    # Test 1D encoder
    encoder_1d, outputs_1d = test_1d_topological_encoder()
    
    # Test 2D encoder
    encoder_2d, outputs_2d = test_2d_topological_encoder()
    
    # Test integration
    topo_encoder, topo_outputs, final_features = integrate_with_existing_pipeline()
    
    # Visualize Hamiltonians
    visualize_hamiltonians()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Integrate TopologicalMLEncoder into your MultiModalMaterialClassifier")
    print("2. Use create_hamiltonian_from_features() to generate Hamiltonians from your existing features")
    print("3. Add topological_features to your fusion network")
    print("4. Train with both main classification loss and topological loss")
    print("5. Analyze local_features to see if the network learns winding angle/Berry curvature")

if __name__ == "__main__":
    main() 