# gpu_spectral_encoder.py
# ──────────────────────────────────────────────────────────────────────────────
"""
GPU-accelerated spectral encoder using PyTorch instead of CPU-intensive SciPy.
This version computes eigenvalues on GPU for much faster training.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian, to_dense_adj
import hashlib

class GPUSpectralEncoder(nn.Module):
    """
    GPU-accelerated spectral encoder with caching and PyTorch-based eigenvalue computation.
    """
    def __init__(self, k_eigs: int, hidden: int, cache_size: int = 100):
        super().__init__()
        self.k = k_eigs
        self.cache_size = cache_size
        self.cache = {}  # Simple LRU cache
        self.mlp = nn.Sequential(
            nn.Linear(k_eigs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def _get_graph_hash(self, edge_index, num_nodes):
        """Create a hash for the graph to enable caching."""
        # Create a simple hash based on graph structure
        edge_hash = hashlib.md5(edge_index.cpu().numpy().tobytes()).hexdigest()
        return f"{edge_hash}_{num_nodes}"

    def _compute_eigenvalues_gpu(self, edge_index, num_nodes, device):
        """GPU-accelerated eigenvalue computation using PyTorch."""
        try:
            # Build normalized Laplacian on GPU
            laplacian_edge_index, laplacian_edge_weight = get_laplacian(
                edge_index, normalization='sym', num_nodes=num_nodes
            )
            
            # Convert to dense adjacency matrix on GPU
            adj = to_dense_adj(laplacian_edge_index, laplacian_edge_weight, 
                              max_num_nodes=num_nodes).squeeze(0).to(device)
            
            # Compute degree matrix
            deg = torch.diag(torch.sum(adj, dim=1))
            
            # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0
            
            L = torch.eye(num_nodes, device=device) - torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
            
            # Add small regularization to ensure positive definiteness
            L = L + 1e-6 * torch.eye(num_nodes, device=device)
            
            # Compute eigenvalues using PyTorch's GPU-accelerated eigendecomposition
            # Use torch.linalg.eigh for symmetric matrices (much faster than eig)
            eigenvals, _ = torch.linalg.eigh(L)
            
            # Sort eigenvalues and take the k smallest non-zero ones
            eigenvals = torch.sort(eigenvals)[0]  # Sort in ascending order
            
            # Skip the first eigenvalue (should be ~0) and take next k
            k_requested = min(self.k, num_nodes - 1)
            if k_requested > 0:
                spec = eigenvals[1:k_requested + 1]
                # Pad with zeros if we don't have enough eigenvalues
                if spec.shape[0] < self.k:
                    padding = torch.zeros(self.k - spec.shape[0], dtype=torch.float, device=device)
                    spec = torch.cat([spec, padding])
            else:
                spec = torch.zeros(self.k, dtype=torch.float, device=device)
            
            return spec
            
        except Exception as e:
            # Fallback: return zeros if computation fails
            print(f"Warning: GPU eigenvalue computation failed: {e}")
            return torch.zeros(self.k, dtype=torch.float, device=device)

    def forward(self, edge_index, num_nodes, batch=None):
        # Handle batch processing
        if batch is not None:
            # Get the number of unique graphs in the batch
            num_graphs = batch.max().item() + 1
            
            # Process a single representative graph for the batch
            single_emb = self._process_single_graph(edge_index, num_nodes)
            
            # Expand to match batch size
            return single_emb.unsqueeze(0).expand(num_graphs, -1)
        else:
            # Single graph case
            return self._process_single_graph(edge_index, num_nodes)
    
    def _process_single_graph(self, edge_index, num_nodes):
        """Process a single graph with GPU-accelerated computation."""
        device = edge_index.device
        
        # Check cache first
        graph_hash = self._get_graph_hash(edge_index, num_nodes)
        if graph_hash in self.cache:
            cached_result = self.cache[graph_hash]
            return cached_result.to(device)
        
        # Compute eigenvalues on GPU
        spec = self._compute_eigenvalues_gpu(edge_index, num_nodes, device)
        
        # Process through MLP (already on correct device)
        result = self.mlp(spec.unsqueeze(0)).squeeze(0)
        
        # Cache the result (simple LRU)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[graph_hash] = result.detach().cpu()
        
        return result

    def clear_cache(self):
        """Clear the cache to free memory."""
        self.cache.clear()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()


class FastSpectralEncoder(nn.Module):
    """
    Ultra-fast spectral encoder using approximate methods for very large graphs.
    """
    def __init__(self, k_eigs: int, hidden: int, cache_size: int = 100):
        super().__init__()
        self.k = k_eigs
        self.cache_size = cache_size
        self.cache = {}
        self.mlp = nn.Sequential(
            nn.Linear(k_eigs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def _get_graph_hash(self, edge_index, num_nodes):
        """Create a hash for the graph to enable caching."""
        edge_hash = hashlib.md5(edge_index.cpu().numpy().tobytes()).hexdigest()
        return f"{edge_hash}_{num_nodes}"

    def _compute_approximate_eigenvalues(self, edge_index, num_nodes, device):
        """Fast approximate eigenvalue computation using power iteration."""
        try:
            # Build adjacency matrix
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
            
            # Compute degree matrix
            deg = torch.sum(adj, dim=1, keepdim=True)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0
            
            # Normalized adjacency matrix: A_norm = D^(-1/2) A D^(-1/2)
            A_norm = torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
            
            # Use power iteration to approximate largest eigenvalues
            # This is much faster than full eigendecomposition
            n = A_norm.shape[0]
            k_requested = min(self.k, n - 1)
            
            if k_requested <= 0:
                return torch.zeros(self.k, dtype=torch.float, device=device)
            
            # Initialize random vectors
            V = torch.randn(n, k_requested, device=device)
            V = torch.nn.functional.normalize(V, dim=0)
            
            # Power iteration
            for _ in range(10):  # 10 iterations usually sufficient
                V_new = torch.mm(A_norm, V)
                V_new = torch.nn.functional.normalize(V_new, dim=0)
                V = V_new
            
            # Compute approximate eigenvalues
            eigenvals = torch.diag(torch.mm(torch.mm(V.T, A_norm), V))
            
            # Convert to Laplacian eigenvalues: λ_L = 1 - λ_A
            laplacian_eigenvals = 1 - eigenvals
            
            # Sort and take the smallest k
            laplacian_eigenvals = torch.sort(laplacian_eigenvals)[0]
            
            spec = laplacian_eigenvals[:k_requested]
            
            # Pad with zeros if needed
            if spec.shape[0] < self.k:
                padding = torch.zeros(self.k - spec.shape[0], dtype=torch.float, device=device)
                spec = torch.cat([spec, padding])
            
            return spec
            
        except Exception as e:
            print(f"Warning: Fast eigenvalue computation failed: {e}")
            return torch.zeros(self.k, dtype=torch.float, device=device)

    def forward(self, edge_index, num_nodes, batch=None):
        device = edge_index.device
        
        # Check cache first
        graph_hash = self._get_graph_hash(edge_index, num_nodes)
        if graph_hash in self.cache:
            cached_result = self.cache[graph_hash]
            return cached_result.to(device)
        
        # Compute approximate eigenvalues
        spec = self._compute_approximate_eigenvalues(edge_index, num_nodes, device)
        
        # Process through MLP
        result = self.mlp(spec.unsqueeze(0)).squeeze(0)
        
        # Cache the result
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[graph_hash] = result.detach().cpu()
        
        return result

    def clear_cache(self):
        """Clear the cache to free memory."""
        self.cache.clear()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache() 