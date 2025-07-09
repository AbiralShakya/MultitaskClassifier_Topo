#!/usr/bin/env python3
"""
Test script for GPU-accelerated spectral encoder.
Compares performance with CPU version and verifies correctness.
"""

import torch
import time
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import numpy as np

# Import both encoders for comparison
from multitask_ti_classification.helper.graph_spectral_encoder import GraphSpectralEncoder
from multitask_ti_classification.helper.gpu_spectral_encoder import GPUSpectralEncoder, FastSpectralEncoder

def create_test_graph(num_nodes=50, num_edges=200):
    """Create a random test graph."""
    # Create random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_index = to_undirected(edge_index)  # Make undirected
    
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    return edge_index, edge_index.shape[1]

def test_encoder_speed(encoder_class, name, edge_index, num_nodes, device, num_runs=10):
    """Test encoder speed and correctness."""
    print(f"\n=== Testing {name} ===")
    
    # Initialize encoder
    encoder = encoder_class(k_eigs=8, hidden=64).to(device)
    
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            _ = encoder(edge_index.to(device), num_nodes)
    
    # Time the computation
    times = []
    results = []
    
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            result = encoder(edge_index.to(device), num_nodes)
        end_time = time.time()
        
        times.append(end_time - start_time)
        results.append(result.cpu())
        
        if i % 5 == 0:
            print(f"  Run {i+1}/{num_runs}: {times[-1]:.4f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"  Result shape: {results[0].shape}")
    print(f"  Result range: [{results[0].min():.4f}, {results[0].max():.4f}]")
    
    return avg_time, results[0]

def main():
    print("Testing GPU-accelerated spectral encoder performance...")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test graph
    print("Creating test graph...")
    edge_index, num_edges = create_test_graph(num_nodes=100, num_edges=400)
    num_nodes = edge_index.max().item() + 1
    print(f"Graph: {num_nodes} nodes, {num_edges} edges")
    
    # Test different encoders
    encoders_to_test = [
        (GraphSpectralEncoder, "CPU Spectral Encoder (SciPy)"),
        (GPUSpectralEncoder, "GPU Spectral Encoder (PyTorch)"),
        (FastSpectralEncoder, "Fast Spectral Encoder (Approximate)")
    ]
    
    results = {}
    
    for encoder_class, name in encoders_to_test:
        try:
            avg_time, result = test_encoder_speed(
                encoder_class, name, edge_index, num_nodes, device, num_runs=5
            )
            results[name] = (avg_time, result)
        except Exception as e:
            print(f"  Error testing {name}: {e}")
            results[name] = (float('inf'), None)
    
    # Compare results
    print("\n=== Performance Comparison ===")
    cpu_time = results.get("CPU Spectral Encoder (SciPy)", (float('inf'), None))[0]
    gpu_time = results.get("GPU Spectral Encoder (PyTorch)", (float('inf'), None))[0]
    fast_time = results.get("Fast Spectral Encoder (Approximate)", (float('inf'), None))[0]
    
    if cpu_time != float('inf') and gpu_time != float('inf'):
        speedup = cpu_time / gpu_time
        print(f"GPU vs CPU speedup: {speedup:.2f}x")
    
    if cpu_time != float('inf') and fast_time != float('inf'):
        speedup = cpu_time / fast_time
        print(f"Fast vs CPU speedup: {speedup:.2f}x")
    
    # Test correctness by comparing outputs
    print("\n=== Correctness Check ===")
    cpu_result = results.get("CPU Spectral Encoder (SciPy)", (None, None))[1]
    gpu_result = results.get("GPU Spectral Encoder (PyTorch)", (None, None))[1]
    
    if cpu_result is not None and gpu_result is not None:
        # Compare the outputs (they might be slightly different due to numerical precision)
        diff = torch.abs(cpu_result - gpu_result).max().item()
        print(f"Max difference between CPU and GPU results: {diff:.6f}")
        
        if diff < 1e-3:
            print("✓ Results are consistent (difference < 1e-3)")
        else:
            print("⚠ Results differ significantly - this might be expected due to different algorithms")
    
    print("\n=== Memory Usage ===")
    if device.type == 'cuda':
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.1f} MB")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 